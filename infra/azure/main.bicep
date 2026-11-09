// One small VM for the public demo: app, Qdrant and Caddy under Docker
// Compose, data on its own managed disk. See docs/DEPLOY.md.
//
// A managed disk rather than Azure Files because SQLite's WAL mode needs
// shared memory, which network file systems do not provide. Only 80 and 443
// are open: deploys and secrets go through the VM run-command API.

@description('Must be in this subscription\'s allowed list: Azure for Students limits regions per account.')
param location string

@description('Globally unique within the region: <label>.<location>.cloudapp.azure.com')
param dnsLabel string

@description('GitHub repository allowed to deploy, as owner/name.')
param githubRepo string

@description('Required by Azure for a Linux VM. Port 22 stays closed.')
param adminPublicKey string

@description('2 vCPU and 4 GiB: the embedding model and Tectonic run in the app container.')
param vmSize string = 'Standard_B2als_v2'

param dataDiskGb int = 32

var name = 'otw-demo'

resource nsg 'Microsoft.Network/networkSecurityGroups@2023-11-01' = {
  name: '${name}-nsg'
  location: location
  properties: {
    securityRules: [
      {
        name: 'allow-web'
        properties: {
          priority: 100
          direction: 'Inbound'
          access: 'Allow'
          protocol: 'Tcp'
          sourceAddressPrefix: 'Internet'
          sourcePortRange: '*'
          destinationAddressPrefix: '*'
          destinationPortRanges: ['80', '443']
        }
      }
    ]
  }
}

resource vnet 'Microsoft.Network/virtualNetworks@2023-11-01' = {
  name: '${name}-vnet'
  location: location
  properties: {
    addressSpace: { addressPrefixes: ['10.20.0.0/24'] }
    subnets: [
      {
        name: 'default'
        properties: {
          addressPrefix: '10.20.0.0/24'
          networkSecurityGroup: { id: nsg.id }
        }
      }
    ]
  }
}

resource ip 'Microsoft.Network/publicIPAddresses@2023-11-01' = {
  name: '${name}-ip'
  location: location
  sku: { name: 'Standard' }
  properties: {
    publicIPAllocationMethod: 'Static'
    dnsSettings: { domainNameLabel: dnsLabel }
  }
}

resource nic 'Microsoft.Network/networkInterfaces@2023-11-01' = {
  name: '${name}-nic'
  location: location
  properties: {
    ipConfigurations: [
      {
        name: 'primary'
        properties: {
          subnet: { id: vnet.properties.subnets[0].id }
          publicIPAddress: { id: ip.id }
        }
      }
    ]
  }
}

resource dataDisk 'Microsoft.Compute/disks@2023-10-02' = {
  name: '${name}-data'
  location: location
  sku: { name: 'StandardSSD_LRS' }
  properties: {
    creationData: { createOption: 'Empty' }
    diskSizeGB: dataDiskGb
  }
}

resource vm 'Microsoft.Compute/virtualMachines@2024-03-01' = {
  name: '${name}-vm'
  location: location
  properties: {
    hardwareProfile: { vmSize: vmSize }
    osProfile: {
      computerName: name
      adminUsername: 'azureuser'
      customData: loadFileAsBase64('cloud-init.yaml')
      linuxConfiguration: {
        disablePasswordAuthentication: true
        ssh: {
          publicKeys: [
            { path: '/home/azureuser/.ssh/authorized_keys', keyData: adminPublicKey }
          ]
        }
      }
    }
    storageProfile: {
      imageReference: {
        publisher: 'Canonical'
        offer: 'ubuntu-24_04-lts'
        sku: 'server'
        version: 'latest'
      }
      osDisk: {
        createOption: 'FromImage'
        diskSizeGB: 30
        managedDisk: { storageAccountType: 'StandardSSD_LRS' }
        deleteOption: 'Delete'
      }
      dataDisks: [
        {
          lun: 0
          createOption: 'Attach'
          managedDisk: { id: dataDisk.id }
          // Deleting the VM must never take the data with it.
          deleteOption: 'Detach'
        }
      ]
    }
    networkProfile: { networkInterfaces: [{ id: nic.id }] }
  }
}

// What GitHub Actions signs in as. No password or key is stored in GitHub.
resource deployer 'Microsoft.ManagedIdentity/userAssignedIdentities@2023-01-31' = {
  name: '${name}-deployer'
  location: location
}

resource federation 'Microsoft.ManagedIdentity/userAssignedIdentities/federatedIdentityCredentials@2023-01-31' = {
  parent: deployer
  name: 'github-main'
  properties: {
    issuer: 'https://token.actions.githubusercontent.com'
    subject: 'repo:${githubRepo}:ref:refs/heads/main'
    audiences: ['api://AzureADTokenExchange']
  }
}

// Virtual Machine Contributor on this one VM, which includes run-command.
// Scoped below the identity, so it cannot edit its own federated credential.
var vmContributor = subscriptionResourceId(
  'Microsoft.Authorization/roleDefinitions',
  '9980e02c-c2be-4d73-94e8-173b1dc7cf3c'
)

resource deployerRole 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(vm.id, deployer.id, vmContributor)
  scope: vm
  properties: {
    roleDefinitionId: vmContributor
    principalId: deployer.properties.principalId
    principalType: 'ServicePrincipal'
  }
}

output publicHost string = ip.properties.dnsSettings.fqdn
output vmName string = vm.name
output deployerClientId string = deployer.properties.clientId
