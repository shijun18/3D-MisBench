
# How to Add a New Model to Our Benchmark
- First, prepare your custom model file `new_model.py`.
- Then, provide a model constructor function in that file that returns the model architecture.
- Write a new model configuration file, following the example of `nnU-Net/nnUNet_new/nnUnet_reconstruction/nnunetv2/mymodel/new_model_cfg.json`, where:
  - `model_path`: is the path to your custom model file mentioned above.
  - `model_create_function_name`: is the name of the function in your model file that returns the model architecture.
  - `model_create_function_interface`: is the usage of your model constructor function. In most cases, this requires the number of segmentation classes num_classes and the number of input image channels in_channel.
  - `model_name`: is the name of the new model you are adding.
  - The following are optional training parameter adjustments, such as `batch_size`, `patch_size`, and `epochs`. (**Note: After you have added your new model, you can find the generated nnUNetTrainer file for your new model in the `nnunetv2/training/nnUNetTrainer` directory. Subsequent adjustments to the model training parameters can be made more conveniently in this file.**)
- Finally, you can use the command `nnUNetv2_add_new_model_code {path_to_your_new_model_configuration_file}` to add the new model.