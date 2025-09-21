# Sample Jenkins server 

* Clone the repository
* `cd jenkins-server`
* `docker compose up -d --build`



## Connect to the server

* `http://localhost:8080`
* in case you need to provide a password
* `docker logs jenkins-blueocean` -> look for 
* An admin user has been created and a password generated.
* Please use the following password to proceed to installation:
* PASSWORD_TO_COPY

* need to provide : 
* admin
* passwd
* full name
* email

## Make jenkins accessible from the Web -> create proxy with ngrok

* install procedure ofr ngrok -> website

* Run
* `ngrok http http://localhost:8080`