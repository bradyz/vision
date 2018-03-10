import torch


def load_by_name(x, name, checkpoint):
    if x is None:
        return

    x.load_state_dict(checkpoint[name])


def maybe_load(state, checkpoint_path):
    print('Loading %s.' % checkpoint_path)

    try:
        checkpoint = torch.load(checkpoint_path)

        for key, value in state.items():
            load_by_name(value, key, checkpoint)

        return True
    except Exception as e:
        print(e)
        print('Failed.')
        pass

    return False


def save(state, checkpoint_path):
    state_dict = {key: value.state_dict() for key, value in state.items()}

    torch.save(state_dict, checkpoint_path)
