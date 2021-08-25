# OpenFraming

## Introduction

We have introduced OpenFraming, a Web-based system for analyzing and classifying frames in the text documents. OpenFraming is designed to lower the barriers to applying machine learning for frame analysis, including giving researchers the capability to build models using their own labeled data. Its architecture is designed to be user-friendly and easily navigable, empowering researchers to com- fortably make sense of their text corpora without specific machine learning knowledge.

## Requirements

### Docker
You need [Docker](https://docs.docker.com/get-docker/). Feel free to read up on Docker if you wish.
Our best short explanation for Docker is that, Docker is for deploying applications with complicated
dependencies, what the printing press was to publishing books (it allows you to do it in a much quicker,
and much more reproducible way).

The link above has guides on how to install Docker on the most popular platforms.

## How to install

 1. `git clone https://github.com/Tdrinker/openFramingSource.git`
 2. `cd openFramingSource`
 3. `docker-compose build`
 4. `docker-compose up`
 
 You might have to add `sudo` at the beginning of commands at step 3 and 4 if using linux/macOS.


## E-mails
If you want to send actual e-mails through Sendgrid with this system (as opposed to just
printing the e-mails that would be sent to the console),  please set the environment
variables:

```bash
export SENDGRID_API_KEY=     # An API key from Sendgrid
export SENGRID_FROM_EMAIL=   # An email address to put in the "from" field. Note that
			     # you'll have to verify this email in Sendgrid as a 
			     # "Sender". 
```

If you happen to need `sudo` in the section above, please pass the `-E` flag to make
sure these environment variables are picked up. i.e.,

```bash
sudo -E docker-compose up
```

## Getting help

If you have any question, concern, or bug report, please file an issue in this repository's Issue Tracker and we will respond accordingly.