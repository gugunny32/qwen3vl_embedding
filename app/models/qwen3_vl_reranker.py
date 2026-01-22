"""
Qwen3-VL Reranker implementation based on official Qwen3-VL-Embedding repository
Source: https://github.com/QwenLM/Qwen3-VL-Embedding
"""

import torch
import numpy as np
from PIL import Image
from typing import List, Union, Optional, Dict
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from loguru import logger


# Default configuration constants
MAX_LENGTH = 10240
IMAGE_BASE_FACTOR = 16
IMAGE_FACTOR = IMAGE_BASE_FACTOR * 2
MIN_PIXELS = 4 * IMAGE_FACTOR * IMAGE_FACTOR
MAX_PIXELS = 1800 * IMAGE_FACTOR * IMAGE_FACTOR
FPS = 1
MAX_FRAMES = 64
FRAME_MAX_PIXELS = 768 * IMAGE_FACTOR * IMAGE_FACTOR
MAX_TOTAL_PIXELS = 10 * FRAME_MAX_PIXELS


def sample_frames(
    frames: List[Union[str, Image.Image]],
    max_segments: int
) -> List[Union[str, Image.Image]]:
    """Sample frames uniformly from a video sequence"""
    duration = len(frames)
    if duration <= max_segments:
        return frames

    frame_id_array = np.linspace(0, duration - 1, max_segments, dtype=int)
    frame_id_list = frame_id_array.tolist()
    sampled_frames = [frames[frame_idx] for frame_idx in frame_id_list]
    return sampled_frames


class Qwen3VLReranker:
    """
    Qwen3-VL Reranker for multimodal reranking
    """
    
    def __init__(
        self,
        model_name_or_path: str,
        max_length: int = MAX_LENGTH,
        min_pixels: int = MIN_PIXELS,
        max_pixels: int = MAX_PIXELS,
        total_pixels: int = MAX_TOTAL_PIXELS,
        fps: float = FPS,
        max_frames: int = MAX_FRAMES,
        default_instruction: str = "Given a search query, retrieve relevant candidates that answer the query.",
        use_cpu: bool = False,  # Force CPU to avoid CUDA issues
        **kwargs,
    ):
        # Use CPU by default to avoid CUDA assertion errors
        self.device = torch.device("cpu" if use_cpu else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.use_cpu = use_cpu
        
        # Force CPU as default device if use_cpu is True
        if use_cpu:
            torch.set_default_device('cpu')
            torch.set_default_dtype(torch.float32)
            logger.info("Set default PyTorch device to CPU")

        self.max_length = max_length
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.total_pixels = total_pixels
        self.fps = fps
        self.max_frames = max_frames
        self.default_instruction = default_instruction

        logger.info(f"Loading Qwen3VLReranker from {model_name_or_path} on {self.device}")

        # Filter out device-related kwargs to ensure CPU usage
        filtered_kwargs = {k: v for k, v in kwargs.items() if k not in ['device', 'device_map', 'torch_dtype']}
        
        # Force CPU dtype when using CPU
        if use_cpu:
            filtered_kwargs['torch_dtype'] = torch.float32
        
        # Load the language model on CPU
        lm = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            device_map='cpu' if use_cpu else 'auto',
            **filtered_kwargs
        )
        
        # Verify model is on correct device
        actual_device = next(lm.parameters()).device
        logger.info(f"Model loaded on device: {actual_device}")

        self.model = lm.model
        self.processor = AutoProcessor.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            padding_side='left'
        )
        self.model.eval()

        # Initialize binary classification head for yes/no scoring
        token_true_id = self.processor.tokenizer.get_vocab()["yes"]
        token_false_id = self.processor.tokenizer.get_vocab()["no"]
        self.score_linear = self.get_binary_linear(lm, token_true_id, token_false_id)
        self.score_linear.eval()
        # Ensure score_linear stays on same device as model
        if use_cpu:
            self.score_linear = self.score_linear.to('cpu').to(torch.float32)
            logger.info(f"Score linear on device: {next(self.score_linear.parameters()).device}")
        else:
            self.score_linear = self.score_linear.to(self.device).to(self.model.dtype)
        
        logger.info(f"Qwen3VLReranker loaded successfully on {self.device}")

    def get_binary_linear(self, model, token_yes: int, token_no: int) -> torch.nn.Linear:
        """Create binary classification layer from yes/no token embeddings"""
        lm_head_weights = model.lm_head.weight.data

        weight_yes = lm_head_weights[token_yes]
        weight_no = lm_head_weights[token_no]

        D = weight_yes.size()[0]
        linear_layer = torch.nn.Linear(D, 1, bias=False)
        with torch.no_grad():
            linear_layer.weight[0] = weight_yes - weight_no
        return linear_layer

    @torch.no_grad()
    def compute_scores(self, inputs: Dict) -> List[float]:
        """Compute relevance scores for query-document pairs"""
        batch_scores = self.model(**inputs).last_hidden_state[:, -1]
        scores = self.score_linear(batch_scores)
        
        # Clamp scores to prevent numerical instability in sigmoid
        scores = torch.clamp(scores, min=-20.0, max=20.0)
        
        scores = torch.sigmoid(scores).squeeze(-1).cpu().detach().tolist()
        
        # Validate scores
        if isinstance(scores, float):
            scores = [scores]
        scores = [max(0.0, min(1.0, s)) for s in scores]  # Ensure in valid range
        
        return scores

    def truncate_tokens_optimized(
        self,
        tokens: List[str],
        max_length: int,
        special_tokens: List[str]
    ) -> List[str]:
        """Truncate tokens while preserving special tokens"""
        if len(tokens) <= max_length:
            return tokens

        special_tokens_set = set(special_tokens)
        num_special = sum(1 for token in tokens if token in special_tokens_set)
        num_non_special_to_keep = max_length - num_special

        final_tokens = []
        non_special_kept_count = 0
        for token in tokens:
            if token in special_tokens_set:
                final_tokens.append(token)
            elif non_special_kept_count < num_non_special_to_keep:
                final_tokens.append(token)
                non_special_kept_count += 1

        return final_tokens

    def tokenize(self, pairs: List[Dict], **kwargs) -> Dict:
        """Tokenize query-document pairs"""
        max_length = self.max_length
        text = self.processor.apply_chat_template(pairs, tokenize=False, add_generation_prompt=True)

        try:
            images, videos, video_kwargs = process_vision_info(
                pairs,
                image_patch_size=16,
                return_video_kwargs=True,
                return_video_metadata=True
            )
        except Exception as e:
            logger.debug(f"No vision info to process: {e}")
            images = None
            videos = None
            video_kwargs = {'do_sample_frames': False}

        if videos is not None:
            videos, video_metadatas = zip(*videos)
            videos, video_metadatas = list(videos), list(video_metadatas)
        else:
            videos, video_metadatas = None, None

        inputs = self.processor(
            text=text,
            images=images,
            videos=videos,
            video_metadata=video_metadatas,
            truncation=False,
            padding=False,
            do_resize=False,
            **video_kwargs
        )

        # Truncate input IDs while preserving special tokens
        for i, ele in enumerate(inputs['input_ids']):
            inputs['input_ids'][i] = self.truncate_tokens_optimized(
                inputs['input_ids'][i][:-5],
                max_length,
                self.processor.tokenizer.all_special_ids
            ) + inputs['input_ids'][i][-5:]

        # Apply padding
        temp_inputs = self.processor.tokenizer.pad(
            {'input_ids': inputs['input_ids']},
            padding=True,
            return_tensors="pt",
            max_length=self.max_length
        )
        for key in temp_inputs:
            inputs[key] = temp_inputs[key]

        return inputs

    def format_mm_content(
        self,
        text: Optional[Union[List[str], str]] = None,
        image: Optional[Union[List[Union[str, Image.Image]], str, Image.Image]] = None,
        video: Optional[Union[List[Union[str, List[Union[str, Image.Image]]]], str, List[Union[str, Image.Image]]]] = None,
        prefix: str = 'Query:',
        fps: Optional[float] = None,
        max_frames: Optional[int] = None,
    ) -> List[Dict]:
        """Format multimodal content into conversation format"""
        content = []
        content.append({'type': 'text', 'text': prefix})

        # Normalize inputs to lists
        texts = [text] if isinstance(text, str) else (text if text else [])
        images = [image] if not isinstance(image, list) and image is not None else (image if image else [])

        # Add text content
        for txt in texts:
            content.append({'type': 'text', 'text': txt})

        # Add image content
        for img in images:
            if isinstance(img, Image.Image):
                image_content = img
            elif isinstance(img, str):
                image_content = img if img.startswith(('http://', 'https://')) else 'file://' + img
            else:
                continue

            content.append({
                'type': 'image',
                'image': image_content,
                "min_pixels": self.min_pixels,
                "max_pixels": self.max_pixels
            })

        return content

    def format_mm_instruction(
        self,
        query_text: Optional[Union[str, tuple]] = None,
        query_image: Optional[Union[List[Union[str, Image.Image]], str, Image.Image]] = None,
        query_video: Optional[Union[List[Union[str, List[Union[str, Image.Image]]]], str, List[Union[str, Image.Image]]]] = None,
        doc_text: Optional[Union[List[str], str]] = None,
        doc_image: Optional[Union[List[Union[str, Image.Image]], str, Image.Image]] = None,
        doc_video: Optional[Union[List[Union[str, List[Union[str, Image.Image]]]], str, List[Union[str, Image.Image]]]] = None,
        instruction: Optional[str] = None,
        fps: Optional[float] = None,
        max_frames: Optional[int] = None
    ) -> List[Dict]:
        """Format multimodal query-document pair for reranking"""
        inputs = []
        inputs.append({
            "role": "system",
            "content": [{
                "type": "text",
                "text": "Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\"."
            }]
        })

        # Handle query_text as tuple containing (instruction, text)
        if isinstance(query_text, tuple):
            instruct, query_text = query_text
        else:
            instruct = instruction

        contents = []
        contents.append({
            "type": "text",
            "text": '<Instruct>: ' + (instruct or self.default_instruction)
        })

        # Format query content
        query_content = self.format_mm_content(
            query_text, query_image, query_video,
            prefix='<Query>:',
            fps=fps,
            max_frames=max_frames
        )
        contents.extend(query_content)

        # Format document content
        doc_content = self.format_mm_content(
            doc_text, doc_image, doc_video,
            prefix='\n<Document>:',
            fps=fps,
            max_frames=max_frames
        )
        contents.extend(doc_content)

        inputs.append({
            "role": "user",
            "content": contents
        })

        return inputs

    def process(
        self,
        inputs: Dict,
    ) -> List[float]:
        """
        Process inputs and generate reranking scores
        
        Args:
            inputs: Dict with keys 'instruction', 'query', 'documents', 'fps', 'max_frames'
        
        Returns:
            List of relevance scores
        """
        instruction = inputs.get('instruction', self.default_instruction)

        query = inputs.get("query", {})
        documents = inputs.get("documents", [])

        if not query or not documents:
            return []

        # Format each query-document pair
        pairs = [
            self.format_mm_instruction(
                query.get('text', None),
                query.get('image', None),
                query.get('video', None),
                document.get('text', None),
                document.get('image', None),
                document.get('video', None),
                instruction=instruction,
                fps=inputs.get('fps', self.fps),
                max_frames=inputs.get('max_frames', self.max_frames)
            )
            for document in documents
        ]

        # Compute scores for each pair
        final_scores = []
        for pair in pairs:
            tokenized_inputs = self.tokenize([pair])
            # Explicitly move to the correct device (CPU if use_cpu was set)
            if self.use_cpu:
                # Force all tensors to CPU
                tokenized_inputs = {k: v.to('cpu') if isinstance(v, torch.Tensor) else v 
                                   for k, v in tokenized_inputs.items()}
            else:
                tokenized_inputs = tokenized_inputs.to(self.device)
            scores = self.compute_scores(tokenized_inputs)
            final_scores.extend(scores)

        return final_scores
