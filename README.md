
In this project, The trained AI will predict the progression of the disease (in my case i took Heart disease)

So, here where i will provide details about my project:

when you open the folder you will see, there are 3 files and a single folder. When you open the folder you will notice that its an excel file with a name heart, That is the file where i used the data to train the AI.

If you want to train my AI in different area (like different disease ) use a dataset releated to it and make sure it is in the excel file if not chnage it to excel file
NOTE:
my AI will take the dataset in the form of a 'EXCEL FILE' only.

Then make sure you copy the path of the dataset file and past it in the file_path = ( past it here )

Now coming back to the main folder where there will be 3 files with the names as heart_disease_model.pkl, python heart_disease_analysis.py, save_model.py and app.py

In python heart_disease_analysis.py have the model where it will show the graphs and data when it is beginnig trained.

The python heart_disease_analysis.pkl is a script and it is created from the file save_model.py

In the save_model.py the model is saved and it creates a pkl file where it will be used to open the Streamlit

and lastly app.py is where the streamlit is

NOTE IMPORTANT:
When you are trying to predict the outcome of a new patient run this commend in the app.py file

THE COMMENND

python -m streamlit run app.py

============================================================================================================================THE END============================================================================================================================================
