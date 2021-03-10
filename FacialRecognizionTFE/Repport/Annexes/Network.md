# Network Configuration

# Network A

--------------------
Plage IP Server
--------------------
192.168.1.0 => 99 : FREE

192.168.1.100 : IPCAM1
192.168.1.101 : IPCAM2
192.168.1.102 : IPCAM3
192.168.1.103 : IPCAM4
192.168.1.104 : IPCAM5
192.168.1.105 : IPCAM6
192.168.1.106 : IPCAM7
192.168.1.107 : Server IPCAM PROD

192.168.1.200 : VMWARE ESXI
192.168.1.201 : VPN + UniFi Controller
192.168.1.202 : Server ZMNINJA
192.168.1.204 : BlackHunter

# Network B

Wireless + Proxymus Tv

192.168.2.0/24

--------------------
Plage IP Server
--------------------

auto eth0
iface eth0 inet static
	address 192.168.1.203
        netmask 255.255.255.0
        gateway 192.168.1.1
