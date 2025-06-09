import os
import tensorflow as tf
import tensorflow_io as tfio
import numpy as np
from typing import Dict, List, Optional, Any, Callable
from openmmrl.utils.logging import get_logger

logger = get_logger(__name__)

class OpenMMRLDatasetTF:
    """
    TensorFlow-based dataset for multimodal data loading and preprocessing.
    """
    
    def __init__(
        self,
        dataset_name: str,
        split: str = "train",
        data_dir: Optional[str] = None,
        modalities: List[str] = ["text", "image"],
        batch_size: int = 32,
        shuffle_buffer_size: int = 10000,
        num_parallel_calls: int = tf.data.AUTOTUNE,
        prefetch_size: int = tf.data.AUTOTUNE,
        text_max_length: int = 512,
        image_size: int = 224,
        audio_sample_rate: int = 16000,
        audio_duration: float = 10.0,
        video_num_frames: int = 32,
    ):
        self.dataset_name = dataset_name
        self.split = split
        self.data_dir = data_dir
        self.modalities = modalities
        self.batch_size = batch_size
        self.shuffle_buffer_size = shuffle_buffer_size
        self.num_parallel_calls = num_parallel_calls
        self.prefetch_size = prefetch_size
        self.text_max_length = text_max_length
        self.image_size = image_size
        self.audio_sample_rate = audio_sample_rate
        self.audio_duration = audio_duration
        self.video_num_frames = video_num_frames
        
        self.dataset = self._create_dataset()
    
    def _create_dataset(self):
        """Create the TensorFlow dataset."""
        # Load raw dataset
        if self.dataset_name.lower() in ["webimagetext", "wit"]:
            dataset = self._load_webimagetext()
        elif self.dataset_name.lower() == "coco":
            dataset = self._load_coco()
        elif self.dataset_name.lower() == "flickr30k":
            dataset = self._load_flickr30k()
        elif self.dataset_name.lower() == "conceptual_captions":
            dataset = self._load_conceptual_captions()
        else:
            # Try loading a custom dataset from data_dir
            dataset = self._load_custom_dataset()
        
        # Apply preprocessing
        dataset = dataset.map(
            self._preprocess_example,
            num_parallel_calls=self.num_parallel_calls
        )
        
        # Filter invalid examples
        dataset = dataset.filter(self._is_valid_example)
        
        # Shuffle if training
        if self.split == "train" and self.shuffle_buffer_size > 0:
            dataset = dataset.shuffle(self.shuffle_buffer_size)
        
        # Batch the dataset
        dataset = dataset.batch(
            self.batch_size,
            drop_remainder=(self.split == "train")
        )
        
        # Prefetch for performance
        dataset = dataset.prefetch(self.prefetch_size)
        
        return dataset
    
    def _load_webimagetext(self):
        """Load Web Image Text dataset."""
        if not self.data_dir:
            raise ValueError("data_dir must be specified for WebImageText dataset")
        
        pattern = os.path.join(self.data_dir, f"wit-{self.split}*.tfrecord")
        filenames = tf.data.Dataset.list_files(pattern, shuffle=(self.split == "train"))
        dataset = tf.data.TFRecordDataset(filenames)
        
        def parse_wit_example(example_proto):
            features = {
                'image/encoded': tf.io.FixedLenFeature([], tf.string),
                'image/format': tf.io.FixedLenFeature([], tf.string),
                'text': tf.io.FixedLenFeature([], tf.string),
                'label': tf.io.FixedLenFeature([], tf.int64, default_value=0),
            }
            return tf.io.parse_single_example(example_proto, features)
        
        return dataset.map(parse_wit_example)
    
    def _load_coco(self):
        """Load COCO dataset."""
        import tensorflow_datasets as tfds
        
        try:
            dataset = tfds.load(
                "coco_captions",
                split=self.split,
                data_dir=self.data_dir,
                with_info=False
            )
            
            def format_coco_example(example):
                return {
                    'image/encoded': tf.image.encode_jpeg(example['image']),
                    'image/format': tf.constant('jpeg'),
                    'text': example['captions']['text'][0],  # Take first caption
                    'label': tf.constant(0, dtype=tf.int64),
                }
            
            return dataset.map(format_coco_example)
            
        except Exception as e:
            logger.error(f"Failed to load COCO dataset: {e}")
            raise
    
    def _load_flickr30k(self):
        """Load Flickr30k dataset."""
        # This would need to be implemented based on your specific Flickr30k format
        raise NotImplementedError("Flickr30k dataset loading not implemented")
    
    def _load_conceptual_captions(self):
        """Load Conceptual Captions dataset."""
        import tensorflow_datasets as tfds
        
        try:
            dataset = tfds.load(
                "conceptual_captions",
                split=self.split,
                data_dir=self.data_dir,
                with_info=False
            )
            
            def format_cc_example(example):
                return {
                    'image/encoded': example['image'],
                    'image/format': tf.constant('jpeg'),
                    'text': example['caption'],
                    'label': tf.constant(0, dtype=tf.int64),
                }
            
            return dataset.map(format_cc_example)
            
        except Exception as e:
            logger.error(f"Failed to load Conceptual Captions dataset: {e}")
            raise
    
    def _load_custom_dataset(self):
        """Load a custom dataset from tfrecord files."""
        if not self.data_dir:
            raise ValueError("data_dir must be specified for custom dataset")
        
        pattern = os.path.join(self.data_dir, f"{self.dataset_name}-{self.split}*.tfrecord")
        filenames = tf.data.Dataset.list_files(pattern, shuffle=(self.split == "train"))
        
        if not filenames:
            raise FileNotFoundError(f"No tfrecord files found matching pattern: {pattern}")
        
        dataset = tf.data.TFRecordDataset(filenames)
        
        def parse_custom_example(example_proto):
            # Define a flexible feature spec
            features = {
                'image/encoded': tf.io.FixedLenFeature([], tf.string, default_value=''),
                'image/format': tf.io.FixedLenFeature([], tf.string, default_value='jpeg'),
                'text': tf.io.FixedLenFeature([], tf.string, default_value=''),
                'audio/encoded': tf.io.FixedLenFeature([], tf.string, default_value=''),
                'video/encoded': tf.io.FixedLenFeature([], tf.string, default_value=''),
                'label': tf.io.FixedLenFeature([], tf.int64, default_value=0),
                'labels': tf.io.VarLenFeature(tf.int64),
            }
            return tf.io.parse_single_example(example_proto, features)
        
        return dataset.map(parse_custom_example)
    
    def _preprocess_example(self, example):
        """Preprocess a single example."""
        result = {}
        
        # Process text
        if "text" in self.modalities and "text" in example:
            text = example["text"]
            if tf.strings.length(text) > 0:
                result["text"] = self._preprocess_text(text)
        
        # Process image
        if "image" in self.modalities and "image/encoded" in example:
            image_encoded = example["image/encoded"]
            if tf.strings.length(image_encoded) > 0:
                result["images"] = self._preprocess_image(image_encoded)
        
        # Process audio
        if "audio" in self.modalities and "audio/encoded" in example:
            audio_encoded = example["audio/encoded"]
            if tf.strings.length(audio_encoded) > 0:
                result["audio"] = self._preprocess_audio(audio_encoded)
        
        # Process video
        if "video" in self.modalities and "video/encoded" in example:
            video_encoded = example["video/encoded"]
            if tf.strings.length(video_encoded) > 0:
                result["video"] = self._preprocess_video(video_encoded)
        
        # Add labels
        if "label" in example:
            result["labels"] = example["label"]
        elif "labels" in example and hasattr(example["labels"], "values"):
            result["labels"] = tf.sparse.to_dense(example["labels"])
        
        return result
    
    def _preprocess_text(self, text):
        """Preprocess text data."""
        # Simple tokenization (you would use a proper tokenizer in practice)
        text = tf.strings.lower(text)
        text = tf.strings.regex_replace(text, r'[^a-zA-Z0-9\s]', '')
        
        # Split into tokens
        tokens = tf.strings.split([text]).values
        
        # Pad or truncate to max_length
        tokens = tokens[:self.text_max_length]
        tokens = tf.concat([tokens, tf.fill([self.text_max_length - tf.shape(tokens)[0]], '')], 0)
        
        # Convert to IDs (placeholder - use proper vocab in practice)
        vocab_table = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(
                keys=tf.constant([''], dtype=tf.string),
                values=tf.constant([0], dtype=tf.int64)
            ),
            default_value=1
        )
        
        input_ids = vocab_table.lookup(tokens)
        attention_mask = tf.cast(tf.strings.length(tokens) > 0, tf.int32)
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
    
    def _preprocess_image(self, image_encoded):
        """Preprocess image data."""
        # Decode image
        image = tf.image.decode_image(image_encoded, channels=3)
        image = tf.cast(image, tf.float32)
        
        # Resize and normalize
        image = tf.image.resize(image, [self.image_size, self.image_size])
        image = tf.image.random_flip_left_right(image) if self.split == "train" else image
        image = image / 255.0
        
        # Normalize with ImageNet stats
        mean = tf.constant([0.485, 0.456, 0.406])
        std = tf.constant([0.229, 0.224, 0.225])
        image = (image - mean) / std
        
        return image
    
    def _preprocess_audio(self, audio_encoded):
        """Preprocess audio data."""
        # Decode audio (assuming WAV format)
        try:
            audio = tfio.audio.decode_wav(audio_encoded, dtype=tf.float32)
            audio = tf.squeeze(audio, axis=-1)  # Remove channel dimension if mono
            
            # Resample to target sample rate if needed
            # This is a placeholder - proper resampling would need librosa or similar
            
            # Pad or truncate to fixed duration
            target_length = int(self.audio_sample_rate * self.audio_duration)
            audio_length = tf.shape(audio)[0]
            
            if audio_length > target_length:
                audio = audio[:target_length]
            else:
                padding = target_length - audio_length
                audio = tf.concat([audio, tf.zeros([padding])], 0)
            
            # Compute mel spectrogram
            stft = tf.signal.stft(
                audio,
                frame_length=1024,
                frame_step=256,
                fft_length=1024
            )
            magnitude = tf.abs(stft)
            
            # Convert to mel scale (simplified)
            mel_features = tf.reduce_mean(magnitude, axis=-1)  # Placeholder
            
            return {
                "features": mel_features,
            }
            
        except Exception as e:
            logger.warning(f"Failed to process audio: {e}")
            # Return dummy features
            target_frames = int(self.audio_sample_rate * self.audio_duration / 256)
            return {
                "features": tf.zeros([target_frames, 513]),
            }
    
    def _preprocess_video(self, video_encoded):
        """Preprocess video data."""
        # This is a placeholder for video preprocessing
        # In practice, you would decode the video and extract frames
        try:
            # Decode video frames (this would need ffmpeg-python or similar)
            # For now, return dummy frames
            dummy_frames = tf.zeros([self.video_num_frames, self.image_size, self.image_size, 3])
            
            return {
                "frames": dummy_frames,
            }
            
        except Exception as e:
            logger.warning(f"Failed to process video: {e}")
            # Return dummy frames
            return {
                "frames": tf.zeros([self.video_num_frames, self.image_size, self.image_size, 3]),
            }
    
    def _is_valid_example(self, example):
        """Check if an example has required modalities."""
        has_required_modality = False
        
        for modality in self.modalities:
            if modality == "text" and "text" in example:
                has_required_modality = True
                break
            elif modality == "image" and "images" in example:
                has_required_modality = True
                break
            elif modality == "audio" and "audio" in example:
                has_required_modality = True
                break
            elif modality == "video" and "video" in example:
                has_required_modality = True
                break
        
        return has_required_modality
    
    def get_dataset(self):
        """Get the preprocessed dataset."""
        return self.dataset 