# Azure ML Documentation
## Terms & Definition
* **Azure ML**: Azure Machine Learning empowers data scientists and developers to build, deploy, and manage high-quality models faster and with confidence. It accelerates time to value with industry-leading machine learning operations (MLOps), open-source interoperability, and integrated tools. 
* **Datastore**: An Azure ML “Datastore” is a reference to an existing storage account on Azure and secures the connection information without risking the authentication credentials and integrity of the source of data.  
* **Dataset**: “Data asset” (also known as a dataset) in Azure Machine Learning studio is a reference to the collection of related data sources, used in various authoring mechanisms like Automated ML, Notebooks, Designer, and Experiments. The data referenced in any Data asset can come from a wide variety of Data Sources like local or web files, Azure blob storage and datastores etc. 
* **Environment**: An Environment is a collection of Python or R packages and libraries, environment variables, and various settings that encapsulate the needs of the machine learning model’s training and scoring scripts.  
* **Compute Instance**: Compute instance is intended to serve as the Machine Learning professional’s development workstation. It’s a VM with multiple tools and environments pre-installed for common machine-learning tasks.  
* **Compute cluster**: Compute cluster is a set of VMs capable of scaling to multiple nodes when needed for large workloads. It scales up automatically when a job is submitted and is well-suited for dev/test deployments. 
* **Experiments**: An experiment is a light-weight container for Run. Use experiments to submit and track runs. An “Experiment” is a collection of multiple runs of a particular training script. 
* **Endpoint**: An endpoint is a stable and durable URL that can be used to request or invoke a model. You provide the required inputs to the endpoint and get the outputs back. Azure Machine Learning allows you to implement online endpoints and batch endpoints. Online endpoints are designed for real-time inference—when you invoke the endpoint, the results are returned in the endpoint's response. Batch endpoints, on the other hand, are designed for long-running batch inference.  
* **Deployment**: A deployment is a set of resources and computes required for hosting the model or component that does the actual inferencing. A single endpoint can contain multiple deployments. Endpoints have a routing mechanism that can direct requests to specific deployments in the endpoint. 

## Process of creating Azure ML Resource

1.	We require a machine learning workspace. Go to Azure Portal and search for Azure Machine Learning 
2.	Create a new Machine Learning Workspace if one doesn’t already exist.  

3.	Fill in the required details, create a storage account, key vault if required. 
4.	The creation of compute, environment and infrastructure is explained later as part of the framework. Codes are present to create these infrastructure. 

## Development using the Framework

We can use the framework to perform Ad Hoc Experimentation and to deploy model end points. We’ll be explaining the approaches below. 

### Ad Hoc Experimentation 

Assuming you already have a Resource Group, Machine Learning workspace and other prerequisites, we can begin with the Ad Hoc experimentation. 
The following steps allow you to execute a Machine Learning (ML) job on Azure ML clusters using the framework. This process does not create an endpoint since this is an Ad Hoc Experimentation. The model will run on the Azure ML compute, and you can monitor its performance and other details in the Azure ML Studio.  
To get the folder structure, we can download it or fork it from the Azure_ML_Template_USPET repo. To execute jobs on the Azure ML workspace via an IDE like PyCharm, you mainly need the **‘src’** folder and a .env file. The **.env**  file should contain all necessary values. **The workspace and datastore should already exist, and their names should be in the .env file.** If the compute and environment do not exist, the code will create them based on the configurations provided in the code. The following sections provide an overview of the folder structure and steps to run your first ML model (Iris Prediction) using this framework. 

#### Step 1: Populating the .env file

* Create a new file named ‘.env’. The file should contain the following information: 
AZURE_WORKSPACE_NAME = "WORKSPACE NAME" 
AZURE_SUBSCRIPTION_ID = "SUBSCRIPTION_ID " 
AZURE_RESOURCE_GROUP = "RESOURCE GROUP NAME" 
AZURE_ML_COMPUTE_TRAINER = "COMPUTE NAME" 
AZURE_ML_EXPERIMENT_NAME = "EXPERIMENT NAME" 
AZURE_ML_DATASTORE_NAME = "DATASTORE NAME" 
AZURE_ML_ENDPOINT_NAME = "ENDPOINT NAME" 
AZURE_ML_ENVIRONMENT_NAME = "ENVIRONMENT NAME" 
AZURE_ML_ENVIRONMENT_VERSION = " ENVIRONMENT VERSION NUMBER" 
* To populate the values, Go to portal.azure.com. Select the Resource Group in which the Azure ML workspace is created. From the essential section, copy the value of Subscription ID and the name of Resource Group itself. In the .env file put this info in the **AZURE_SUBSCRIPTION_ID, AZURE_RESOURCE_GROUP** fields respectively. Also fill the name of the Azure ML workspace in the **AZURE_WORKSPACE_NAME** field. 
* Open the Azure ML workspace, at the top in the ‘Essentials’, click on the Storage. This opens the Storage Account. From the menu on the left, select ‘Containers’. Create a new container here and provide a name. For this example, we’re using the name ‘flatfilestorage’. This place will store your input files and model output files. We’ll link this storage to our Azure ML studio later. In our case we are running the Iris classification problem and will be storing our input dataset ‘Iris_Dataset.csv’ here. 
* From the Azure ML Workspace, click on “Launch Studio”, from the menu on left select ‘Compute’. Go to ‘Compute Clusters’. If the compute already exists, put the compute name in **AZURE_ML_COMPUTE_TRAINER** field. If the compute doesn’t exist, we can manually create it here or the script will create it for us using whatever name you pass in the **AZURE_ML_COMPUTE_TRAINER** field. 
* In the AML studio, under ‘Assets’ select ‘Data’. In here, select ‘Datastores’ and create a new ‘Datastore’. 

Give it a name, select your Subscription_id, Storage account and Blob Container. In our case we named it ‘flatfilestorage’ so we’ll select it.Finally, you need to provide the **Account Key** and create a connection with the container you created in step 2. 
To get the account key, go to the storage account & select ‘Access key’ under the ‘Security+Networking’ tab. Click in ‘Show’ & copy the value for Key. Paste this value in the ‘Account Key’ field when creating datastore in the AML Studio. 

Once done, you’ll be able to see the ‘Iris_Dataset.csv’ in the AML Studio datastore as well. 
Use the name of this newly created datastore (AML Studio) in the .env file in the field **AZURE_ML_DATASTORE_NAME**. 
 
•	We now have all the required environment variables. Similarly, we can fill in values for other variables like **‘AZURE_ML_ENVIRONMENT_NAME’, ‘AZURE_ML_ENVIRONMENT_VERSION’**. 
If you have already created the environment, put the name of the existing environment. Assuming this is the first time we are setting up this framework and no such environment exists, still provide the name and the ‘main.py’ file will create the environment for you with the same name. 

#### Step 2: Understanding the folder structure & main.py working

We’ll mainly be using the **‘src -> azureml_v2 -> batch_single_step’** folder for our ad hoc experimentation.   
Our **‘main.py’** is the driver file which takes in the arguments – **‘input data path’** & **‘output data path’**. These is also a default argument ‘config-path’. 
* **Input data path**: Defines the location of the input file inside the datastore. Earlier we already defined our datastore name (**AZURE_ML_DATASTORE_NAME**) in the .env file. This Input data path gets appended to the datastore name and a full input file path is created.Since we kept our data set **‘Iris_Dataset.csv’** just inside the Datastore/Blob Storage (at the root) we just write the name of the file. 
* **Output data path**: Define the location in which you want to store the output of the model. For now we’ll just be storing it in the root of our Datastore, so we just pass ‘/’. 
* **Config-path**: This argument already has a default value **‘./cfg/config.yaml’**. This file is used to define the compute and environment parameters and used to create the infrastructure when it does not exist. 
 
We need to add arguments to the main.py file so right click on the ‘main.py’ file and select ‘modify run configuration’. In the parameters field paste : 	 
**--input-data-path "/iris_dataset.csv" --output-data-path "/"** 


* This will act as the input parameters for your script. 
* The main.py file tries to find a compute with the same name (in AML Workspace) as the **‘AZURE_ML_COMPUTE_TRAINER’** value present in the .env file and if it exists, it uses that compute. If the compute doesn't exist it creates one with the same name & specifications. 
* Then the main.py file tries to find an environment with the same name (in AML workspace) as ‘AZURE_ML_ENVIRONMENT_NAME’ value present in the .env file. If the environment exists, we use it else a new environment with the same name is created. 
* **‘cfg’** folder contains the ‘config.yaml’ file which also acts as a default argument for our ‘main.py’ script. This file contains the specifications of compute & environment. This file is used when our compute and environment doesn’t exists and we have to create a new one. Our compute and environment will be created based on whatever specifications we add in this file. This file also refers to the environment folder 
* The **‘environment’** folder contains the ‘conda.yaml’ file which contains all the dependencies and libraries required to run our source code in the AML environment. The environment should have all these libraries installed to run our code. We should change the dependencies in this file based on the codes present in the **‘job’** folder. 
* The **‘job’** folder contains all the code related to our Machine learning model. This is what gets sent to our AML Workspace to run. The ‘run.py’ file inside the ‘job’ folder contains the python code for our Machine Learning Model. 
* The **‘utils’** folder has two files. The ‘settings.py’ file helps check if we are running the code locally or through a build agent & also helps create a ML client. The ‘validation.py’ helps check if the conda dependencies are the same in the local environment and the AML environment. 

#### Step 3: Execution – Quick Steps 

* In local PyCharm, create a new environment, open the template folder structure & install the following libraries :  
a.	Omegaconf 
b.	azure-ai-ml 
c.	azure-cli 
d.	python-dotenv 
* Set up the .env file as shown previously. 
* Add the input parameters to the ‘main.py’ file. 
* Run the command **‘az login’** in terminal for authentication. Select your account in the browser. 


* Run the ‘main.py’ file. 
* Wait for the code to run. Click on the link generated to open the AML workspace and monitor the job. 


* Once the model runs successfully, we can check the metrics, logs and other info in the AML Workspace.

###	Creating & Deploying Models with CI/CD pipelines 
#### Step 1: Understanding the CI/CD codes 
 
At the root of the folder structure, we’ll see the **“devops_pipelines”** folder. This folder contains all the CI, CD pipeline codes along with some supporting template codes and variables required to run the pipelines. 
**Training-pipeline-ci** : This is the CI pipeline script which runs every time we push a change to our branch. Based on which branch we are running this pipeline from; it loads the required variables. E.g. if we are running our CI process for the development branch, it loads the variables from the dev.yaml file. If we are running it for the production branch, it loads the variables from the prod.yaml file. 
In this we can add our various CI checks and steps. The CI pipeline itself calls a template file named **“code-quality-ci.yaml”**, which checks the formatting of our codes using Flake8 & Black Formatter libraries and installs the required dependencies 
 
**Training-pipeline-cd** : This pipeline code also loads variables from dev.yaml or prod.yaml based on which branch we are running the code for. It calls the template file named “publish-aml-endpoint-cd” and provides parameters for the script to run. In the **Training-pipeline-cd.yaml** file we’ll see that while calling the “publish-aml-endpoint-cd.yaml” script we provide parameters as shown below.  
```python
azure_workspace_name: ${{variables.AZURE_WORKSPACE_NAME}} 
azure_resource_group: ${{variables.AZURE_RESOURCE_GROUP}} 
azure_subscription_id: ${{variables.AZURE_SUBSCRIPTION_ID}} 
azure_ml_compute_trainer : ${{variables.AZURE_ML_COMPUTE_TRAINER}}
```
 
These variables like “AZURE_WORKSPACE_NAME” are coming from the dev.yaml/prod.yaml file so we need to make sure whatever variables we pass should be present in the variable files.  
The “publish-aml-endpoint-cd” script installs the required dependencies, gets the Sp Id, Key and using the input parameters, runs the main.py present in the “src/azureml_v2/pipeline_two_step/main.py”, creates the model and an endpoint. 
 
**Variable Files (dev.yaml, prod.yaml)**: These files contain all the variables required to run the pipelines and codes. It contains all the information like RG name, Compute Name, Location of the script to run and other environment variables.  
If running it in production, we wish to run it on a different compute or different environment we just need to change the prod.yaml file.  
 
 
 
#### Step 2:  Understanding the src folder 
 
The src/azureml_v2/pipeline_two_step folder contains all the codes related to creating the model, setting up the environments, deploying the model etc.  
 
*	The **‘cfg’** folder contains the ‘config.yaml’ file which helps define the specification for the compute creation if it doesn’t already exist. It also refers to the ‘conda.yaml’ file present in the environment folder to install the required libraries in the environment to run the codes. 
*	The **‘Job’** folder contains the codes related to the model. The subfolder **‘cfg’** contains the ‘train_config.yaml’ file in which we can define our model parameters. The subfolder **‘utils’** contains supporting scripts for the model like scripts for data preprocessing. Th Job folder also contains modules for data ingestion from the datastore/location & for predictions.  
*	The **‘utils’** folder, contains the ‘settings.py’ file which loads the variables based on if are using a build agent to run the codes or if we are running it locally  
* **‘Main.py’** is the script which gets run from the cd pipeline and essentially contains the whole process of creating infrastructure, running the model and deploying it. It tests if the compute and the environment as mentioned in the variables are present. If not, it creates the compute and environment for us. It creates a pipeline in Azure ML where steps like Ingestion of Data, splitting of data into test train, and prediction is done. It also involves steps like
  * Creating a batch endpoint
  * Creating a pipeline component
  * Create a batch deployment
  * Invoking a deployment 

#### Step 3: Execution 
 
Ideally every different project should have a different repo. Fork the contents for the template repo and make changes. Then push these changed codes to the pre-development branch of your project repo. After raising PR we can push the codes to our development and master branch. 
 
To run the pipelines: 
1.  Go to Azure Devops. 
2.	Navigate to Pipelines and create a new pipeline using Azure Repos Git 
3.	Select your Repo and select ‘Existing Azure Pipelines YAML Files’ 
4.	We’ll set up CI pipeline first, so we select our branch and the training-pipeline-ci file.  

