from ray.rllib.agents import ppo

def show_model_statistics(config: dict) -> None:
    
    # get model from ray
    dummy_trainer = ppo.PPOTrainer(config)
    torch_model = dummy_trainer.get_policy().model
    # show torch model statistics
    total_params = sum(param.numel() for param in torch_model.parameters())
    trainable_params = sum(p.numel() for p in torch_model.parameters() if p.requires_grad)
    print(torch_model)
    print(f"Number of TOTAL parameters: {total_params}")
    print(f"Number of TRAINABLE parameters: {trainable_params}")
    

    #save model in ONNX
    save_path = "dev_and_test/saved_models"
    dummy_trainer.export_policy_model(save_path)
