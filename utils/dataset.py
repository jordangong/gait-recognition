import os
import random
import re
from typing import Optional, Dict, NewType, Union, List, Set

import numpy as np
import torch
from torch.utils import data
from torchvision.io import read_image
import torchvision.transforms as transforms

ClipLabels = NewType('ClipLabels', Set[str])
ClipConditions = NewType('ClipConditions', Set[str])
ClipViews = NewType('ClipViews', Set[str])

default_frame_transform = transforms.Compose([
    transforms.Resize(size=(64, 32))
])


class CASIAB(data.Dataset):
    """CASIA-B multi-view gait dataset"""

    def __init__(
            self,
            root_dir: str,
            is_train: bool = True,
            train_size: int = 74,
            num_sampled_frames: int = 30,
            selector: Optional[Dict[
                str, Union[ClipLabels, ClipConditions, ClipLabels]
            ]] = None,
            num_input_channels: int = 3,
            frame_height: int = 64,
            frame_width: int = 32,
            device: torch.device = torch.device('cpu')
    ):
        """
        :param root_dir: Directory to dataset root.
        :param is_train: Train or test, True for train, False for test.
        :param train_size: Number of subjects in train, when `is_train`
            is False, test size will be inferred.
        :param num_sampled_frames: Number of sampled frames for train
        :param selector: Restrict data labels, conditions and views
        :param num_input_channels Number of input channel, RBG image
            has 3 channel, grayscale image has 1 channel
        :param frame_height Frame height after transforms
        :param frame_width Frame width after transforms
        :param device Device be used for transforms
        """
        super(CASIAB, self).__init__()
        self.root_dir = root_dir
        self.is_train = is_train
        self.train_size = train_size
        self.num_sampled_frames = num_sampled_frames
        self.num_input_channels = num_input_channels
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.device = device

        self.frame_transform: transforms.Compose
        transform_compose_list = [
            transforms.Resize(size=(self.frame_height, self.frame_width))
        ]
        if self.num_input_channels == 1:
            transform_compose_list.insert(0, transforms.Grayscale())
        self.frame_transform = transforms.Compose(transform_compose_list)

        # Labels, conditions and views corresponding to each video clip
        self.labels: np.ndarray[np.str_]
        self.conditions: np.ndarray[np.str_]
        self.views: np.ndarray[np.str_]
        # Video clip directory names
        self._clip_names: List[str] = []
        # Labels, conditions and views in dataset,
        #   set of three attributes above
        self.metadata = Dict[str, Set[str]]

        clip_names = sorted(os.listdir(self.root_dir))

        if self.is_train:
            clip_names = clip_names[:self.train_size * 10 * 11]
        else:  # is_test
            clip_names = clip_names[self.train_size * 10 * 11:]

        # Remove empty clips
        for clip_name in clip_names.copy():
            if len(os.listdir(os.path.join(self.root_dir, clip_name))) == 0:
                print("Clip '{}' is empty.".format(clip_name))
                clip_names.remove(clip_name)

        # clip name constructed by label, condition and view
        # e.g 002-bg-02-090 means clip from Subject #2
        #     in Bag #2 condition from 90 degree angle
        labels, conditions, views = [], [], []
        if selector:
            selected_labels = selector.pop('labels', None)
            selected_conditions = selector.pop('conditions', None)
            selected_views = selector.pop('views', None)

            label_regex = r'\d{3}'
            condition_regex = r'(nm|bg|cl)-0[0-4]'
            view_regex = r'\d{3}'

            # Match required data using RegEx
            if selected_labels:
                label_regex = '|'.join(selected_labels)
            if selected_conditions:
                condition_regex = '|'.join(selected_conditions)
            if selected_views:
                view_regex = '|'.join(selected_views)
            clip_regex = '(' + ')-('.join([
                label_regex, condition_regex, view_regex
            ]) + ')'

            for clip_name in clip_names:
                match = re.fullmatch(clip_regex, clip_name)
                if match:
                    labels.append(match.group(1))
                    conditions.append(match.group(2))
                    views.append(match.group(3))
                    self._clip_names.append(match.group(0))

            self.metadata = {
                'labels': selected_labels,
                'conditions': selected_conditions,
                'views': selected_views
            }
        else:  # Add all
            self._clip_names += clip_names
            for clip_name in self._clip_names:
                split_clip_name = clip_name.split('-')
                label = split_clip_name[0]
                labels.append(label)
                condition = '-'.join(split_clip_name[1:2 + 1])
                conditions.append(condition)
                view = split_clip_name[-1]
                views.append(view)

        self.labels = np.asarray(labels)
        self.conditions = np.asarray(conditions)
        self.views = np.asarray(views)

        if not selector:
            self.metadata = {
                'labels': set(self.labels.tolist()),
                'conditions': set(self.conditions.tolist()),
                'views': set(self.views.tolist())
            }

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> Dict[str, Union[str, torch.Tensor]]:
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

    def _read_video(self, clip_name: str) -> torch.Tensor:
        frames = []
        clip_path = os.path.join(self.root_dir, clip_name)
        sampled_frame_names = self._sample_frames(clip_path)
        for frame_name in sampled_frame_names:
            frame_path = os.path.join(clip_path, frame_name)
            frame = read_image(frame_path)
            frame = self.frame_transform(frame.to(self.device))
            frames.append(frame.cpu())
        clip = torch.stack(frames)

        return clip

    def _sample_frames(self, clip_path: str) -> List[str]:
        frame_names = os.listdir(clip_path)
        if self.is_train:
            num_frames = len(frame_names)
            if num_frames < self.num_sampled_frames:
                frame_names = random.choices(frame_names,
                                             k=self.num_sampled_frames)
            else:
                frame_names = random.sample(frame_names,
                                            k=self.num_sampled_frames)

        return sorted(frame_names)
