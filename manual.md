##  AWS/Docker instructions:  

#### Handy instructions for docker for the next person who wants to take over :
Go to the project directory, then:
-  `sudo docker-compose build`, then `sudo docker-compose up` if building this project for the first time
- If already built for the first time, use `sudo docker-compose up --build` for one step building and running
- `sudo docker-compose logs` for looking at the most recent logs of al the services
No need to go to project directory
- `sudo docker ps` for looking at what services are running
- `sudo docker kill <process ID>` to stop a services (get process id from docker ps)
- if your system is running out of space, `sudo docker system prune -a`
- to stop docker from running on linux: `sudo service docker stop`, to start: `sudo docker service start`, look at status: `sudo service docker status`
- to look at logs of a specific service(ex- if you want to see how web service or classifier is doing) : `sudo docker logs -t -f <process Name>` (last column in docker ps), pressing `ctrl+c` to close logs won't close the service so this is not harmful

##### Where all the data related to db and classifiers and topic models is stored:
for linux, all the data is stored in : 
`~/var/lib/docker/volumes`
- metadata.db is our database which contains all information(cannot delete)
- `openframing_project_data_volume/_data` will contain all the classifiers and topic models(cannot delete)
-  `openframing_transformers_cache_volume` contains cache(downloaded transformer)(can delete will not affect anything)
These 3 files/folders will stay even if you do docker system prune command, you have to manually delete them, just delete the volumes folder and create a new empty volumes folder ---> if doing this, also delete and replace image, containers and network folder --->do this all after `sudo service docker stop` ---> after deleting `sudo service docker start`

#### AWS instructions for whoever takes over:
- you will need an ssh key to ssh into aws server (will give that to professor derry)
- add special permissions to it `chmod 400 key_name.pem`
- `ssh -i "<key_name>.pem" ubuntu@ec2-3-90-135-165.compute-1.amazonaws.com` --> command to ssh into aws (also available in ec2 menu/your instance/connect if you login to aws)
- once logged in type `exit` to quit session
- to copy something from your computer to aws ---> `scp -i "<key_name>.pem" <your file location> ubuntu@ec2-3-90-135-165.compute 1.amazonaws.com:<location in aws>`

### Github
- the current running openframing version is in `vibhu/openFraming folder` so make all the changes here and run docker-compose here.
- use the branch `aws-server` on github and merge all your changes to that branch.
- currently master and ui-revamp and aws-server are the most latest branches so make sure your branches have these as base.
    - Edit in a new branch -> merge to master -> merge to aws-server
- If you have a new changes on github and want to get them on aws, go to : vibhu/openFraming folder
    - `git stash`
    - `git fetch origin`
    - `git merge origin/aws-server`
    - open .env file (`nano .env`)
        - enter sendgrid API key and email(look at the sendgrid API section)
    - docker-compose (look at docker instructions section)

#### Sendgrid API :
API key : 
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
sendgrid account:
    email: help.openframing@gmail.com
    pass: HelpIsOpenFraming23
#### Google :
email : help.openframing@gmail.com
pass: HelpIsOpenFraming.22

(This credential info also in google keep for help.openframing account)