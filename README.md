# Final-project

This repo try to solve the Kaggle competition in 2022, [UW-Madison GI Tract Image Segmentation](https://www.kaggle.com/competitions/uw-madison-gi-tract-image-segmentation/overview). 

To get the data for the different notebooks, we use the Kaggle API to download the data for this competition. 

Create an account on Kaggle. Once logged in, go to the "Rules" page for the Cats vs. Dogs competition and accept the rules by clicking on the button. (required for downloading the dataset)
The following two questions assume you are on Google Colab. If running locally, instead follow the instructions on Kaggle's API documentation and create the required folders yourself.

On Kaggle, go to "Account > Create API Token" on the site, to download the API Token file kaggle.json. Upload this file to the Google Colab runtime, move it to the directory ~/.kaggle and set its permissions using the code below.
