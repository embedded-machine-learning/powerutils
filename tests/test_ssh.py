import paramiko

ssh_ip = "192.168.1.13"
ssh_username = "cdleml"
ssh_command_stress = "stress -c 6 -t 1 -v"
ssh_command_runOnTX2 = "/media/cdleml/128GB/Users/lsteindl/masterthesis/runOnTx2_reduced.sh"

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.client.AutoAddPolicy)
client.connect(ssh_ip, username=ssh_username)
stdin, stdout, stderr = client.exec_command(ssh_command_runOnTX2)

for line in stdout:
    try:
        print('... ' + line.strip('\n'))
    except KeyboardInterrupt:
        print("exited on line:", line)
client.close()