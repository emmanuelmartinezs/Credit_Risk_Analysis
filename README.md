# Credit Risk Analysis

## Overview of Project
Jill commends you for all your hard work. Piece by piece, you’ve been building up your skills in data preparation, statistical reasoning, and machine learning. You are now ready to apply machine learning to solve a real-world challenge: credit card risk.

Credit risk is an inherently unbalanced classification problem, as good loans easily outnumber risky loans. Therefore, you’ll need to employ different techniques to train and evaluate models with unbalanced classes. Jill asks you to use `imbalanced-learn` and `scikit-learn` libraries to build and evaluate models using resampling.

Using the credit card credit dataset from LendingClub, a peer-to-peer lending services company, you’ll oversample the data using the `RandomOverSampler` and `SMOTE` algorithms, and undersample the data using the `ClusterCentroids` algorithm. Then, you’ll use a combinatorial approach of over and undersampling using the `SMOTEENN` algorithm. Next, you’ll compare two new machine learning models that reduce bias, `BalancedRandomForestClassifier` and `EasyEnsembleClassifier`, to predict credit risk. Once you’re done, you’ll evaluate the performance of these models and make a written recommendation on whether they should be used to predict credit risk.

## Deliverables:
This new assignment consists of three technical analysis deliverables and a written report.

1. ***Deliverable 1:*** Use Resampling Models to Predict Credit Risk
2. ***Deliverable 2:*** Use the SMOTEENN Algorithm to Predict Credit Risk
3. ***Deliverable 3:*** Use Ensemble Classifiers to Predict Credit Risk
4. ***Deliverable 4:*** A Written Report on the Credit Risk Analysis [README.md](https://github.com/emmanuelmartinezs/Credit_Risk_Analysis)


## Deliverables:
This new assignment consists of three technical analysis deliverables and a proposal for further statistical study:

* Data Source: ` Module-17-Challenge-Resources.zip` and `LoanStats_2019Q1.csv`
* Data Tools:  `credit_risk_resampling_starter_code.ipynb` and `credit_risk_ensemble_starter_code.ipynb`.
* Software: `Python 3.9`, `Visual Studio Code 1.50.0`, `Anaconda 4.8.5`, `Jupyter Notebook 6.1.4` and `Pandas`


## Resources and Before Start Notes:

![logo](https://github.com/emmanuelmartinezs/Credit_Risk_Analysis/blob/main/Resources/Images/Header.jpg?raw=true)


### Supervised Machine Learning and Credit Risk
#### Predicting Credit Risk

**Create a Machine Learning Environment**

Your new virtual environment will use Python 3.7 and accompanying Anaconda packages. After creating the new virtual environment, you'll install the imbalanced-learn library in that environment.

NOTE
Consult the imbalanced-learn documentation (Links to an external site.) for additional information about the imbalanced-learn library.

Check out the macOS instructions below, or jump to the Windows instructions.

macOS Setup
Before we create a new environment in macOS, we'll need to update the global conda environment:

If your PythonData environment is activated when you launch the command line, deactivate the environment.

REWIND
To deactivate an active environment, type conda deactivate.

Update the global conda environment by typing conda update conda and press Enter.

After all the packages are collected, you'll see the prompt Proceed ([y]/n)?. Press the "Y" key (for "yes") and press Enter.

In the command line, type conda create -n mlenv python=3.7 anaconda. The name of your new environment is mlenv.

After all the packages are collected, you'll see the prompt Proceed ([y]/n)?. Press the "Y" key (for "yes") and press Enter.

Activate your mlenv environment by typing conda activate mlenv and press Enter.

Check Dependencies for the imbalanced-learn Package
Before we install the imbalanced-learn package, we need to confirm that all of the package dependencies are satisfied in our mlenv environment:

NumPy, version 1.11 or later
SciPy, version 0.17 or later
Scikit-learn, version 0.21 or later


On the command line, you can check all packages that begin with numpy, scipy, and scikit-learn when you type conda list | grep and press Enter. The grep command will search for patterns of the text numpy in our conda list. For example, when we type conda list | grep numpy and press Enter, the output should be as follows:

data-17-1-1-1-Condal-List-Grep-Numpy.png

As you can see, our numpy dependency meets the installation requirements for the imbalanced-learn package.

Additionally, you can type python followed by the command argument -c, and then "import `package_name`;print(`package_name`.__version__)" to verify which version of a package is installed in an environment, where `package_name` is the name of the package you want to verify.

Type python -c "import numpy ;print(numpy.__version__)" and then press Enter to see the version of numpy in your mlenv environment.

Windows Setup
Before we create a new environment in Windows, we'll need to update the global conda environment:

Launch the Anaconda Prompt, or open your PythonData Anaconda Prompt and deactivate this environment.

REWIND
To deactivate an active environment, type conda deactivate.

Update the global conda environment by typing conda update conda and press Enter

After all the packages are collected, you'll see the prompt Proceed ([y]/n)?. Press the "Y" key (for "yes") and press Enter.

In the command line, type conda create -n mlenv python=3.7 anaconda.

After all the packages are collected, you'll see the prompt Proceed ([y]/n)?. Press the "Y" key (for "yes") and press Enter.

Activate your mlenv environment by typing conda activate mlenv and press Enter, or open your Anaconda Prompt (mlenv).

Check Dependencies for the imbalanced-learn Package
Before we install the imbalanced-learn package, we need to confirm that all of the package dependencies are satisfied in our mlenv environment:

NumPy, version 1.11 or later
SciPy, version 0.17 or later
Scikit-learn, version 0.21 or later


In the Anaconda Prompt, you can check all packages that begin with numpy, scipy, and scikit-learn when you type conda list | findstr and press Enter. The findstr command will search for patterns of the text in our conda list. For example, when we type conda list | findstr numpy and press Enter, the output should be as follows:

When you run the conda list | findstr numpy, the output terminal
shows NumPy version 1.16.5 is
installed.

From the output, we can see that our numpy dependency meets the installation requirements for the imbalanced-learn package.

Additionally, you can type python followed by the command argument -c, and then "import `package_name`;print(`package_name`.__version__)" to verify which version of a package is installed in an environment, where `package_name` is the name of the package you want to verify:

Type python -c "import numpy;print(numpy.__version__)" and press Enter to see the version of numpy in your mlenv environment.


Install the imbalanced-learn Package
Now that our dependencies have been met, we can install the imbalanced-learn package in our mlenv environment.

With the mlenv environment activated, either in the Terminal in macOS or in the Anaconda Prompt (mlenv) in Windows, type the following:

conda install -c conda-forge imbalanced-learn

Then press Enter.

After all the packages are collected, you'll see the prompt Proceed ([y]/n)?. Press the "Y" key (for "yes") and press Enter.



Add the Machine Learning Environment to Jupyter Notebook
To use the mlenv environment we just created in the Jupyter Notebook, we need to add it to the kernels. In the command line, type python -m ipykernel install --user --name mlenv and press Enter.

To check if the mlenv is installed, launch the Jupyter Notebook and click the "New" dropdown menu:

Click the "New" button to see the mlenv environment in the dropdown
menu.

Now we can begin our machine learning journey.

> Let's move on!

# Deliverable 1:  
## Perform ETL on Amazon Product Reviews 
### Deliverable Requirements:

Using the cloud ETL process, you’ll create an AWS RDS database with tables in pgAdmin, pick a dataset from the [Amazon Review datasets](https://s3.amazonaws.com/amazon-reviews-pds/tsv/index.txt), and extract the dataset into a DataFrame. You'll transform the DataFrame into four separate DataFrames that match the table schema in pgAdmin. Then, you'll upload the transformed data into the appropriate tables and run queries in pgAdmin to confirm that the data has been uploaded.

> To Deliver. 

**Follow the instructions below:**

1. From the following [Amazon Review datasets](https://s3.amazonaws.com/amazon-reviews-pds/tsv/index.txt), pick a dataset that you would like to analyze. All the datasets have the same schemata, as shown in this image:

![d1](https://github.com/emmanuelmartinezs/Amazon_Vine_Analysis/blob/main/Resources/Images/1.png)

2. Create a new database with Amazon RDS just as you did in this module.

3. In pgAdmin, create a new database in your Amazon RDS server that you just create.

4. Download the `challenge_schema.sql` file to your computer.

5. In pgAdmin, run a new query to create the tables for your new database using the code from the `challenge_schema.sql` file.

- After you run the query, you should have the following four tables in your database: customers_table, products_table, review_id_table, and vine_table.

6. Download the `Amazon_Reviews_ETL_starter_code.ipynb` file, then upload the file as a Google Colab Notebook, and rename it `Amazon_Reviews_ETL`.

**NOTE**
> If you try to open the `Amazon_Reviews_ETL_starter_code.ipynb` with jupyter notebook it will give you an error.

7. First **extract** one of the review datasets, then create a new DataFrame.
8. Next, follow the steps below to **transform** the dataset into four DataFrames that will match the schema in the pgAdmin tables:

**NOTE**
> Some datasets have a large number of rows, which will affect the time it takes to complete the following steps.

**The customers_table DataFrame**
To create the `customers_table`, use the code in the `Amazon_Reviews_ETL_starter_code.ipynb` file and follow the steps below to aggregate the reviews by `customer_id`.

* Use the `groupby()` function on the customer_id column of the DataFrame you created in Step 6.
* Count all the customer ids using the `agg()` function by chaining it to the `groupby()` function. After you use this function, a new column will be created, `count(customer_id)`.
* Rename the `count(customer_id)` column using the `withColumnRenamed()` function so it matches the schema for the `customers_table` in pgAdmin.
* The final `customers_table` DataFrame should look like this:

![d1](https://github.com/emmanuelmartinezs/Amazon_Vine_Analysis/blob/main/Resources/Images/2.png)


**The products_table DataFrame**
To create the `products_table`, use the `select()` function to select the `product_id` and `product_title`, then drop duplicates with the `drop_duplicates()` function to retrieve only unique `product_ids`. Refer to the code snippet provided in the `Amazon_Reviews_ETL_starter_code.ipynb` file for assistance.

The final `products_table` DataFrame should look like this:

![d1](https://github.com/emmanuelmartinezs/Amazon_Vine_Analysis/blob/main/Resources/Images/3.png)

**The review_id_table DataFrame**
To create the `review_id_table`, use the `select()` function to select the columns that are in the `review_id_table` in [pgAdmin](https://docs.github.com/en/github/authenticating-to-github/removing-sensitive-data-from-a-repository), and convert the review_date column to a date using the code snippet provided in the `Amazon_Reviews_ETL_starter_code.ipynb` file.

The final `review_id_table` DataFrame should look like this:

![d1](https://github.com/emmanuelmartinezs/Amazon_Vine_Analysis/blob/main/Resources/Images/4.png)

**The vine_table DataFrame**
To create the `vine_table`, use the `select()` function to select only the columns that are in the `vine_table` in [pgAdmin](https://docs.github.com/en/github/authenticating-to-github/removing-sensitive-data-from-a-repository).

The final `vine_table` DataFrame should look like this:

![d1](https://github.com/emmanuelmartinezs/Amazon_Vine_Analysis/blob/main/Resources/Images/5.png)

**Load the DataFrames into pgAdmin**
1. Make the connection to your AWS RDS instance.
2. Load the DataFrames that correspond to tables in pgAdmin.
3. In pgAdmin, run a query to check that the tables have been populated.

**IMPORTANT**
> Before uploading anything to GitHub be sure to remove all sensitive information such as passwords and connection strings. If you have accidentally done so already see this link (Links to an external site.) for more information.

When you’re done, export your `Amazon_Reviews_ETL` Google Colab Notebook as an ipynb file, and save it to your Amazon_Vine_Analysis GitHub repository.

**NOTE**
Uploading each DataFrame can take up to 10 minutes or longer, so it’s a good idea to double-check your work before uploading. If you have problems uploading your work, you may have to shut down the pgAdmin server and restart. Alternatively, you may have to delete the tables and create them again, then re-run your `Amazon_Reviews_ETL` Google Colab Notebook.

**IMPORTANT**
Be sure that you don’t leave your RDS instance up too long. Try to get all your work for Deliverable 1 done in one sitting, then shut down your instance. Please consult the AWS clean-up videos for more information about shutting down your RDS instance. You will not be graded on anything contained strictly in your RDS, so be sure to shut it down.

#### Deliverable 1 Requirements
You will earn a perfect score for Deliverable 1 by completing all requirements below:

* The `Amazon_Reviews_ETL.ipynb` file does the following:
    * An Amazon Review dataset is extracted as a DataFrame
    * The extracted dataset is transformed into four DataFrames with the correct columns
    * All four DataFrames are loaded into their respective tables in pgAdmin



### DELIVERABLE RESULTS:

**Helpful Reviews (All) with 5 Star:**  
For all reviews and "helpful" reviews, **around half of the ratings are 5 Star**, which indicates that the Vine programs tend to give 5 Stars over any other rating.
![d1](https://github.com/emmanuelmartinezs/Amazon_Vine_Analysis/blob/main/Resources/Images/r5.png)

**Percentage of Vine Reviews are 5-star:**  
For all the Vine Reviews, we found almost the same, a little more lower ratings than 5 Star.
![d1](https://github.com/emmanuelmartinezs/Amazon_Vine_Analysis/blob/main/Resources/Images/r6.png)  


**Percentage of Non-Vine Reviews are 5-star:** 
In General, the non-Vine reviews is higher of 5 Stars on non-Vine reviews than 5 Star Vine.  
![d1](https://github.com/emmanuelmartinezs/Amazon_Vine_Analysis/blob/main/Resources/Images/r7.png)

**Vine Review vs. Non-Vine Review**:   
For the entire Furniture product review file, the majority has a small Amazon Vine review:   
![d1](https://github.com/emmanuelmartinezs/Amazon_Vine_Analysis/blob/main/Resources/Images/r1.png)

Now, applying the same analysis over smaller dataset, with "helpful" reviews, we faound an average percentage from the Vine program:  
![d1](https://github.com/emmanuelmartinezs/Amazon_Vine_Analysis/blob/main/Resources/Images/r2.png)

**5 Star Reviews Vine vs Non-Vine:** 
For the entire review dataset, we found a small 5 Star reviews from Vine reviews, **around 0.3%**  
![d1](https://github.com/emmanuelmartinezs/Amazon_Vine_Analysis/blob/main/Resources/Images/r3.png)

By Filtering the "helpful" reviews only, we saw and found a light difference; **a lower 1%** of the 5 Star review from Vine.  
![d1](https://github.com/emmanuelmartinezs/Amazon_Vine_Analysis/blob/main/Resources/Images/r4.png)


### SUMMARY


1. The majority of reviews for Furniture product are almost nothing or lower results from Vine participants: **99.6% are Non-Vine**.  
2. And overall of all 5 Star reviews are also the same as the Furniture, all are from Vine participants: **99.7% of all 5-star reviews are non-Vine**.
3. But we need to highlight that not all of the 5 Star reviews are coming from Vine participants.


### RECOMMENDATIONS:
Below some recommendations to follow:

1. The Amazon Vine Analysis provide a favorable dataset on the 5-star rating.

2. In addition, we found that much data isn't Vine reviews over specific products, that we could minimize the resluts and create a different dataset on just Vine products.

> In addition, 

The analysis gave us that **1/4 are Vine Reviews**
  
![d1](https://github.com/emmanuelmartinezs/Amazon_Vine_Analysis/blob/main/Resources/Images/r8.png)  

Specific Product provide an average of **57% 5 Star reviews**  

![d1](https://github.com/emmanuelmartinezs/Amazon_Vine_Analysis/blob/main/Resources/Images/r9.png)  

For the majority of Vine Reviews, the analysis provide a **49% of 5 Star reviews**   

![d1](https://github.com/emmanuelmartinezs/Amazon_Vine_Analysis/blob/main/Resources/Images/r10.png)   

And for the majority of the non-Vine Reviews, the analysis provide a **60% of 5 Star reviews**

![d1](https://github.com/emmanuelmartinezs/Amazon_Vine_Analysis/blob/main/Resources/Images/r11.png) 




##### Furniture Products - Credit Risk Analysis Completed by Emmanuel Martinez
