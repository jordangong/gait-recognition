import os
import random
import re
from typing import Optional, NewType, Union, List, Tuple, Set, Dict

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from torch.utils import data
from tqdm import tqdm

ClipClasses = NewType('ClipClasses', Set[str])
ClipConditions = NewType('ClipConditions', Set[str])
ClipViews = NewType('ClipViews', Set[str])


class CASIAB(data.Dataset):
    """CASIA-B multi-view gait dataset"""

    def __init__(
            self,
            root_dir: str,
            is_train: bool = True,
            train_size: int = 74,
            num_sampled_frames: int = 30,
            discard_threshold: int = 15,
            selector: Optional[Dict[
                str, Union[ClipClasses, ClipConditions, ClipViews]
            ]] = None,
            num_input_channels: int = 3,
            frame_size: Tuple[int, int] = (64, 32),
            cache_on: bool = False
    ):
        """
        :param root_dir: Directory to dataset root.
        :param is_train: Train or test, True for train, False for test.
        :param train_size: The number of subjects used for training,
            when `is_train` is False, test size will be inferred.
        :param num_sampled_frames: The number of sampled frames.
            (Training Only)
        :param discard_threshold: Discard the sample if its number of
            frames is less than this threshold.
        :param selector: Restrict output data classes, conditions and
            views.
        :param num_input_channels: The number of input channel(s),
            RBG image has 3 channels, grayscale image has 1 channel.
        :param frame_size: Frame height and width after transforming.
        :param cache_on: Preload all clips in memory or not, this will
            increase loading speed, but will add a preload process and
            cost a lot of RAM. Loading the entire dataset
            (is_train = True, train_size = 124, discard_threshold = 1,
            num_input_channels = 3, frame_height = 64, frame_width = 32)
            need about 22 GB of RAM.
        """
        super().__init__()
        self._root_dir = root_dir
        self._is_train = is_train
        self._num_sampled_frames = num_sampled_frames
        self._cache_on = cache_on

        self._frame_transform: transforms.Compose
        transform_compose_list = [
            transforms.Resize(size=frame_size),
            transforms.ToTensor()
        ]
        if num_input_channels == 1:
            transform_compose_list.insert(0, transforms.Grayscale())
        self._frame_transform = transforms.Compose(transform_compose_list)

        # Labels, conditions and views corresponding to each video clip
        self.labels: np.ndarray[np.int64]
        self.conditions: np.ndarray[np.str_]
        self.views: np.ndarray[np.str_]
        # Labels, classes, conditions and views in dataset,
        #   set of three attributes above
        self.metadata: Dict[str, List[np.int64, str]]

        # Dictionaries for indexing frames and frame names by clip name
        # and chip path when cache is on
        self._cached_clips_frame_names: Optional[Dict[str, List[str]]] = None
        self._cached_clips: Optional[Dict[str, torch.Tensor]] = None

        # Video clip directory names
        self._clip_names: List[str] = []
        clip_names = sorted(os.listdir(self._root_dir))

        if self._is_train:
            clip_names = clip_names[:train_size * 10 * 11]
        else:  # is_test
            clip_names = clip_names[train_size * 10 * 11:]

        # Remove clips under threshold
        discard_clips_names = []
        for clip_name in clip_names.copy():
            clip_path = os.path.join(self._root_dir, clip_name)
            if len(os.listdir(clip_path)) < discard_threshold:
                discard_clips_names.append(clip_name)
                clip_names.remove(clip_name)
        if len(discard_clips_names) != 0:
            print(', '.join(discard_clips_names[:-1]),
                  'and', discard_clips_names[-1], 'will be discarded.')

        # Clip name constructed by class, condition and view
        # e.g 002-bg-02-090 means clip from Subject #2
        #     in Bag #2 condition from 90 degree angle
        classes, conditions, views = [], [], []
        if selector:
            selected_classes = selector.pop('classes', None)
            selected_conditions = selector.pop('conditions', None)
            selected_views = selector.pop('views', None)

            class_regex = r'\d{3}'
            condition_regex = r'(nm|bg|cl)-0[0-6]'
            view_regex = r'\d{3}'

            # Match required data using RegEx
            if selected_classes:
                class_regex = '|'.join(selected_classes)
            if selected_conditions:
                condition_regex = '|'.join(selected_conditions)
            if selected_views:
                view_regex = '|'.join(selected_views)
            clip_re = re.compile('(' + ')-('.join((
                class_regex, condition_regex, view_regex
            )) + ')')

            for clip_name in clip_names:
                match = clip_re.fullmatch(clip_name)
                if match:
                    classes.append(match.group(1))
                    conditions.append(match.group(2))
                    views.append(match.group(3))
                    self._clip_names.append(match.group(0))

        else:  # Add all
            self._clip_names += clip_names
            for clip_name in self._clip_names:
                split_clip_name = clip_name.split('-')
                class_ = split_clip_name[0]
                classes.append(class_)
                condition = '-'.join(split_clip_name[1:2 + 1])
                conditions.append(condition)
                view = split_clip_name[-1]
                views.append(view)

        # Encode classes to labels
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(classes)
        self.labels = self.label_encoder.transform(classes)
        self.conditions = np.asarray(conditions)
        self.views = np.asarray(views)

        self.metadata = {
            'labels': list(dict.fromkeys(self.labels.tolist())),
            'classes': self.label_encoder.classes_.tolist(),
            'conditions': list(dict.fromkeys(self.conditions.tolist())),
            'views': list(dict.fromkeys(self.views.tolist()))
        }

        if self._cache_on:
            self._cached_clips_frame_names = dict()
            self._cached_clips = dict()
            self._preload_all_video()

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(
            self,
            index: int
    ) -> Dict[str, Union[np.int64, str, torch.Tensor]]:
        label = self.labels[index]
        condition = self.conditions[index]
        view = self.views[index]
        clip_name = self._clip_names[index]
        clip = self._read_video(clip_name)
        sample = {
            'label': label,
            'condition': condition,
            'view': view,
            'clip': clip
        }

        return sample

    def _preload_all_video(self):
        for clip_name in tqdm(self._clip_names,
                              desc='Preloading dataset', unit='clips'):
            self._read_video(clip_name, is_caching=True)

    def _read_video(self, clip_name: str,
                    is_caching: bool = False) -> torch.Tensor:
        clip_path = os.path.join(self._root_dir, clip_name)
        sampled_frame_names = self._sample_frames(clip_path, is_caching)

        if self._cache_on:
            if is_caching:
                clip = self._read_frames(clip_path, sampled_frame_names)
                self._cached_clips[clip_name] = clip
            else:  # Load cache
                cached_clip = self._cached_clips[clip_name]
                # Return full clips while evaluating
                if not self._is_train:
                    return cached_clip
                cached_clip_frame_names \
                    = self._cached_clips_frame_names[clip_path]
                # Index the original clip via sampled frame names
                clip = self._load_cached_video(cached_clip,
                                               cached_clip_frame_names,
                                               sampled_frame_names)
        else:  # Cache off
            clip = self._read_frames(clip_path, sampled_frame_names)

        return clip

    def _load_cached_video(
            self,
            clip: torch.Tensor,
            frame_names: List[str],
            sampled_frame_names: List[str]
    ) -> torch.Tensor:
        # Mask the original clip when it is long enough
        if len(frame_names) >= self._num_sampled_frames:
            sampled_frame_mask = np.isin(frame_names,
                                         sampled_frame_names)
            sampled_clip = clip[sampled_frame_mask]
        else:  # Create a indexing filter from the beginning of clip
            sampled_index = frame_names.index(sampled_frame_names[0])
            sampled_frame_filter = [sampled_index]
            for i in range(1, self._num_sampled_frames):
                if sampled_frame_names[i] != sampled_frame_names[i - 1]:
                    sampled_index += 1
                sampled_frame_filter.append(sampled_index)
            sampled_clip = clip[sampled_frame_filter]

        return sampled_clip

    def _read_frames(self, clip_path, frame_names):
        frames = []
        for frame_name in frame_names:
            frame_path = os.path.join(clip_path, frame_name)
            frame = Image.open(frame_path)
            frame = self._frame_transform(frame)
            frames.append(frame)
        clip = torch.stack(frames)

        return clip

    def _sample_frames(self, clip_path: str,
                       is_caching: bool = False) -> List[str]:
        if self._cache_on:
            if is_caching:
                # Sort frame in advance for loading convenience
                frame_names = sorted(os.listdir(clip_path))
                self._cached_clips_frame_names[clip_path] = frame_names
                # Load all without sampling
                return frame_names
            else:  # Load cache
                frame_names = self._cached_clips_frame_names[clip_path]
        else:  # Cache off
            frame_names = os.listdir(clip_path)

        if self._is_train:
            num_frames = len(frame_names)
            # Sample frames without replace if have enough frames
            if num_frames < self._num_sampled_frames:
                frame_names = random.choices(frame_names,
                                             k=self._num_sampled_frames)
            else:
                frame_names = random.sample(frame_names,
                                            k=self._num_sampled_frames)

        return sorted(frame_names)
