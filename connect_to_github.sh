#!/bin/bash

eval "$(ssh-agent -s)"
ssh-add ~/.ssh/github_arafat_shovon
ssh -T git@github.com

