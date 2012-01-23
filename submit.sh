#!/bin/bash

# If you wish to hardcode your username into the script, replace the arg number 
# with a string and decrement the other arg numbers
# Ex: USERNAME="dciliske"
# PASSWORD=$1
USERNAME=ffoo
PASSWORD=Ae,kei.jd13
FILENAME=$1
curl -F "MAX_FILE_SIZE=" -F "uploadedfile=@$FILENAME" -F "accountname=$USERNAME" -F "password=$PASSWORD" -F "_submit_check=1" http://www.dci.pomona.edu/tools-bin/cs131upload.php