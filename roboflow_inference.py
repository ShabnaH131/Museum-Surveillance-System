from inference_sdk import InferenceHTTPClient

client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="vtf3CfW6Qh9cOG8gI0ca"
)

result = client.run_workflow(
    workspace_name="museum-surveillance-system",
    workflow_id="find-bottles",
    images={
        "image": "YOUR_IMAGE.jpg"
    },
    use_cache=True # cache workflow definition for 15 minutes
)