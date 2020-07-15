#Constant value for build, run and deployment
image_name=experiment-dashboard
user=ianpegg9
cluster_name=emotion-api-cluster-nginx
num_nodes=1
zone=us-central1-b
port=80
version=latest
updated_version=latest

#Build and run docker image on local
deploy-local:
	#build docker image
	docker build -t ${user}/${image_name}:${version} .
	#run docker image
	docker run -d --name ${image_name} -p $(port):$(port) ${user}/${image_name}:${version}

#Build and deploy docker image on kubernetes GCP
deploy-gcp:
	#Build the container image of face emotion application and tag it for uploading
	docker build -t gcr.io/${user}/${image_name}:${version} .
	#Using the gcloud command line tool, install the Kubernetes command-line tool
	#kubectl is used to communicate with Kubernetes, which is the cluster orchestration system of GKE clusters
	docker push gcr.io/${user}/${image_name}:${version}
	#Use gcloud command-line tool set the project id:
	gcloud config set project ${user}
	#Use gcloud command-line tool set the zone:
	gcloud config set compute/zone ${zone}
	#Create a one-node kubernetes cluster named emotion-api-cluster-nginx on GCP
	gcloud container ${cluster_name} create face-emotion --num-nodes=${num_nodes}
	#Deploy face emotion application, listening on port 80:
	kubectl run ${image_name} --image=gcr.io/${user}/${image_name}:v1 --port 80
	#Expose face emotion application to traffic from the Internet
	kubectl expose deployment ${image_name} --type=LoadBalancer --port $(port) --target-port $(port)

#Deploy a new version of app
update-deploy-new-version:
	#Create an image for the v2 version of face emotion application by building the same source code and tagging it as v2
	docker build -t gcr.io/${user}/${image_name}:${updated_version} .
	#Push the image to the Google Container Registry
	gcloud docker -- push gcr.io/${user}/${image_name}:${updated_version}
	#Apply a rolling update to the existing deployment with an image update
	kubectl set image deployment/${image_name} ${image_name}=gcr.io/${user}/${image_name}:${updated_version}

#Destroy the service and kubernetes cluster from gcp
destroy:
	#Delete the Service
	kubectl delete deployment ${image_name}
