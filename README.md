# serverless-llm-demo

This project demonstrates how to build a serverless LLM inference API using AWS Lambda with SnapStart, FastAPI, and llama-cpp-python. It includes the following components:

- `app/` - FastAPI application code for the Lambda function
- `layers/llama-cpp/` - Custom Lambda layer containing llama-cpp-python and its dependencies
- `models/` - Directory for local model files
- `template.yaml` - SAM template defining the application's AWS resources

## Features

- FastAPI-based Lambda function for LLM inference
- Lambda SnapStart for improved cold starts
- Custom Lambda layer with llama-cpp-python
- Qwen 1.5B model quantized to 4-bit for efficient inference
- Uses memfd to overcome Lambda storage size limitations for large model files
- AWS Lambda Web Adapter integration for running FastAPI and enabling response streaming

## Prerequisites

To deploy this application, you need:

* [AWS SAM CLI](https://docs.aws.amazon.com/serverless-application-model/latest/developerguide/serverless-sam-cli-install.html)
* [Python 3.12](https://www.python.org/downloads/)
* [Docker](https://hub.docker.com/search/?type=edition&offering=community)

## Deployment

1. Download the Qwen model and set up S3:
   ```bash
   # Download the Qwen model
   wget https://huggingface.co/bartowski/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/Qwen2.5-1.5B-Instruct-Q4_K_M.gguf
   
   # Create an S3 bucket (replace BUCKET_NAME with your desired bucket name)
   aws s3 mb s3://BUCKET_NAME
   
   # Upload the model to S3
   aws s3 cp Qwen2.5-1.5B-Instruct-Q4_K_M.gguf s3://BUCKET_NAME/
   ```

2. Build and deploy the application using SAM CLI:
   ```bash
   # Build the application
   sam build --use-container

   # Deploy the application (update MODEL_BUCKET parameter with your bucket name)
   sam deploy --guided
   ```

During the guided deployment, you'll be prompted for:

* **Stack Name**: Name for your CloudFormation stack
* **AWS Region**: Target region for deployment
* **MODEL_BUCKET**: Name of the S3 bucket containing your model file
* **Confirm changes before deploy**: Option to review changes before deployment
* **Allow SAM CLI IAM role creation**: Required for creating necessary IAM roles
* **Save arguments to samconfig.toml**: Save settings for future deployments

After deployment, SAM will output the Function URL for your API endpoint.

## Architecture

The application consists of:

1. A FastAPI application running on Lambda that handles LLM inference requests
2. A custom Lambda layer containing llama-cpp-python compiled for AWS Lambda
3. Lambda SnapStart enabled for faster cold starts
4. AWS Lambda Web Adapter for FastAPI integration and response streaming
5. S3 access to load the Qwen model with memfd for efficient model loading

## Local Development

To run the application locally:

1. Install dependencies:
```bash
cd app
pip install -r requirements.txt
```

2. Download a compatible GGUF model file to the models directory

3. Run the FastAPI application:
```bash
cd app
uvicorn main:app --reload
```

## Cleanup

To remove the deployed application:

```bash
sam delete
```

## Resources

- [AWS Lambda SnapStart](https://docs.aws.amazon.com/lambda/latest/dg/snapstart.html)
- [AWS Lambda Function URLs](https://docs.aws.amazon.com/lambda/latest/dg/lambda-urls.html)
- [AWS Lambda Web Adapter](https://github.com/awslabs/aws-lambda-web-adapter)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [llama-cpp-python](https://github.com/abetlen/llama-cpp-python)
