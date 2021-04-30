#!/bin/bash
set -e

os_major_version=$(cat /etc/redhat-release | tr -dc '0-9.'|cut -d \. -f1)


if ! rpm -q --quiet epel-release ; then
  yum install -y https://dl.fedoraproject.org/pub/epel/epel-release-latest-$os_major_version.noarch.rpm
fi

echo "installing for os major version : $os_major_version"
yum install -y which gdb redhat-lsb-core expat-devel libcurl-devel tar unzip curl zlib-devel make libunwind icu aria2 rsync bzip2 git bzip2-devel

if [ "$os_major_version" == "7" ]; then
    # install dotnet core dependencies
    yum install -y lttng-ust openssl-libs krb5-libs libicu libuuid
    # install dotnet runtimes
    yum install -y https://packages.microsoft.com/config/centos/7/packages-microsoft-prod.rpm
    yum install -y dotnet-sdk-2.1
fi

yum install -y java-11-openjdk-devel

#If the /opt/python folder exists, we assume this is the manylinux docker image
if [ ! -d "/opt/python/cp37-cp37m" ]; then
  yum install -y ccache gcc gcc-c++ python3 python3-devel python3-pip
fi


if [ ! -d "/usr/local/cuda-10.2" ]; then
    echo "Installing GCC 10"
cat <<EOF > /etc/yum.repos.d/oracle.repo
[oracle_software_collections]
name=Software Collection packages for Oracle Linux 7 (\$basearch)
baseurl=http://yum.oracle.com/repo/OracleLinux/OL7/SoftwareCollections/\$basearch/
gpgkey=file:///etc/pki/rpm-gpg/RPM-GPG-KEY-oracle
gpgcheck=1
enabled=1
EOF


curl -L -o /etc/pki/rpm-gpg/RPM-GPG-KEY-oracle https://yum.oracle.com/RPM-GPG-KEY-oracle-ol7 

gpg --import /etc/pki/rpm-gpg/RPM-GPG-KEY-oracle

yum remove -y devtoolset-*-binutils devtoolset-*-gcc devtoolset-*-gcc-c++ devtoolset-*-gcc-gfortran
yum install -y devtoolset-10-binutils devtoolset-10-gcc devtoolset-10-gcc-c++ devtoolset-10-gcc-gfortran
fi
