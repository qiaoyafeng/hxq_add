from models.multi_model import get_models

visual_net, audio_net = get_models()

print(f"visual_net: {visual_net}")
print(f"audio_net: {audio_net}")
