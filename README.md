# Connecting to the GPU Cluster

 1. **Download the VPN**
https://vpngate.ntu.edu.sg/global-protect/getsoftwarepage.esp
2. **(If not on NTUSecure) Connect to the VPN**
- login use network password (same as NTU learn)
- set up 2FA by scanning QR code with a authenticator app on your phone
3. **(Optional) Generate a SSH key** 
- generate key so that when connecting in the future there is no need to enter your password
- KeyPhrase can be left blank i.e `-N ""`
```
# on local
ssh-keygen -t ecdsa -b 521 -C "<comments>" -f ~/.ssh/<keyFileName> -N "<keyPhrase>"
```
4. **SSH to GPU cluster head node**
- connect and enter password when prompted
```
on local
ssh -p 22 <networkID>@<serverIP>
```
- if you enter an incorrect password multiple times (3?) you may have to wait a while to try again

5. **(Optional) update `~/.ssh/authorized_keys`**

- create `~/.ssh` directory if it doesn't exist
```
# on remote, if ~/.ssh doesn't exist
mkdir ~/.ssh 
chmod 700 ~/.ssh
```
- create `~/.ssh/authorized_keys` file if it doesn't exist
```
# on remote if ~/.ssh/authorized_keys doesn't exist
touch ~/.ssh/authorized_keys
chmod 600 ~/.ssh/authorized_keys
```
- copy content of public key file to `~/.ssh/authorized_keys`
```
# on remote
echo "<publicKeyFileContents>" >> ~/.ssh/authorized_keys
```

6. **(optional) Update /.ssh/config**
- update the ssh config file for simpler connecting
```
# on local
# add the following to ~/.ssh/config

host <alias> 						  # with alias of your choice, e.g. NTU_GPU
	Hostname <serverIP> 				  #ip of NTU GPU Cluster
	IdentityFile ~/.ssh/<KeyFileName> # path to your ssh private key file
	Port 22	
	User <networkID>                  # Network account ID	
```

- now you can connect using:
```
ssh <alias>
```

# SSH Tunneling
To access jupyter notebook service or other remote services, you can do local port forwarding.
```
# on remote
juypter lab
```
- take note of the port in the url it gives 
- e.g. `http://localhost:8890/lab?token=<token>`

```
# on local
ssh -L <localport>:localhost:<remoteport> -p 22 <networkId>:<serverIP>

# if you modified .ssh/config to allow ssh with an alias
ssh -L <localport>:localhost:<remoteport> <alias>
```
- to connect to `http://localhost:8890/lab?token=<token>`, enter `http://localhost:<localport>/lab?token=<token>' on your local browser

# Running SLURM jobs on GPU nodes
refer to https://www3.ntu.edu.sg/scse/fyp/UsefulInfo/SCSEGPU-TC1-UG-UserGuide.pdf



