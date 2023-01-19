follow the steps for execute the denoising process.

1. train the model

   run the following script in terminal at the root directory

   ```shell
   sh script/train_denoise.sh
   ```

2. add noise to the ground truth

   run the following script in terminal at the root directory

   ```shell
   python add_noise
   ```

3. run restoration program

   run the following script in terminal at the root directory

   ```shell
   python my_eval.py
   ```

   remember to change the code of 101 to

   ```python
   utils.load_checkpoint(model_restoration,'<relative path of the model>')
   ```

   

