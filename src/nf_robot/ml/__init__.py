import importlib
import importlib.abc
import importlib.util
import sys

# Old flat names of the modules now in nf_robot.ml.lerobot, kept importable.
_MOVED_TO_LEROBOT = {
    'stringman_lerobot': 'stringman',
    **{f'lerobot_{name}': name for name in (
        'build_dataset',
        'derive_dataset',
        'eval_modal',
        'expand_smolvla_state_dim',
        'find_frozen_video',
        'label_contact_actions',
        'move_version_tag',
        'normalize_tasks',
        'reblend_ortho',
        'repair_episode_meta',
        'resize_video_feature',
        'split_dataset',
        'train_modal',
        'trim_to_grasp',
    )},
}


class _MovedModuleLoader(importlib.abc.Loader):
    def __init__(self, new_name):
        self.new_name = new_name

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        # The import system returns whatever sys.modules holds once this returns.
        sys.modules[module.__name__] = importlib.import_module(self.new_name)

    def get_code(self, fullname):
        # Used by `python -m <old name>`, which runs the code rather than importing it.
        spec = importlib.util.find_spec(self.new_name)
        return spec.loader.get_code(self.new_name)


class _MovedModuleFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        parent, _, name = fullname.rpartition('.')
        if parent != __name__ or name not in _MOVED_TO_LEROBOT:
            return None
        new_name = f'{__name__}.lerobot.{_MOVED_TO_LEROBOT[name]}'
        origin = importlib.util.find_spec(new_name).origin
        return importlib.util.spec_from_loader(fullname, _MovedModuleLoader(new_name), origin=origin)


sys.meta_path.append(_MovedModuleFinder())
