// Resources inside the MatchSense resource group. Everything is keyless: the app signs in to the
// registry, Foundry and Application Insights with its own managed identity.
targetScope = 'resourceGroup'

param location string
param tags object
param resourceToken string
param principalId string
param modelName string
param modelVersion string
param modelCapacity int
param useFoundryAgents bool

var abbrs = {
  logs: 'log-'
  insights: 'appi-'
  identity: 'id-'
  registry: 'cr'
  foundry: 'aif-'
  project: 'proj-'
  environment: 'cae-'
  app: 'ca-'
}
var foundryName = '${abbrs.foundry}${resourceToken}'
var projectName = '${abbrs.project}matchsense'
var modelDeploymentName = modelName

// Built-in role IDs: https://learn.microsoft.com/azure/role-based-access-control/built-in-roles
var acrPullRoleId = subscriptionResourceId('Microsoft.Authorization/roleDefinitions', '7f951dda-4ed3-4680-a7ca-43fe172d538d')
var azureAiUserRoleId = subscriptionResourceId('Microsoft.Authorization/roleDefinitions', '53ca6127-db72-4b80-b1b0-d745d6d5456d')
var metricsPublisherRoleId = subscriptionResourceId('Microsoft.Authorization/roleDefinitions', '3913510d-42f4-4e42-8a64-420c390055eb')

// ------------------------------------------------------------------ monitoring

resource workspace 'Microsoft.OperationalInsights/workspaces@2025-02-01' = {
  name: '${abbrs.logs}${resourceToken}'
  location: location
  tags: tags
  properties: {
    sku: { name: 'PerGB2018' }
    retentionInDays: 30
    features: { enableLogAccessUsingOnlyResourcePermissions: true }
  }
}

resource appInsights 'Microsoft.Insights/components@2020-02-02' = {
  name: '${abbrs.insights}${resourceToken}'
  location: location
  kind: 'web'
  tags: tags
  properties: {
    Application_Type: 'web'
    WorkspaceResourceId: workspace.id
    DisableLocalAuth: true
  }
}

// -------------------------------------------------------------------- identity

resource appIdentity 'Microsoft.ManagedIdentity/userAssignedIdentities@2024-11-30' = {
  name: '${abbrs.identity}matchsense-${resourceToken}'
  location: location
  tags: tags
}

// -------------------------------------------------------------------- registry

resource registry 'Microsoft.ContainerRegistry/registries@2025-05-01-preview' = {
  name: '${abbrs.registry}${resourceToken}'
  location: location
  tags: tags
  sku: { name: 'Basic' }
  properties: {
    adminUserEnabled: false
    anonymousPullEnabled: false
    publicNetworkAccess: 'Enabled'
    networkRuleBypassOptions: 'AzureServices'
  }
}

resource acrPull 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(registry.id, appIdentity.id, acrPullRoleId)
  scope: registry
  properties: {
    roleDefinitionId: acrPullRoleId
    principalId: appIdentity.properties.principalId
    principalType: 'ServicePrincipal'
  }
}

// --------------------------------------------------------------------- Foundry

resource foundry 'Microsoft.CognitiveServices/accounts@2026-07-01' = {
  name: foundryName
  location: location
  tags: tags
  kind: 'AIServices'
  sku: { name: 'S0' }
  identity: { type: 'SystemAssigned' }
  properties: {
    allowProjectManagement: true
    customSubDomainName: foundryName
    disableLocalAuth: true
    publicNetworkAccess: 'Enabled'
  }
}

resource foundryProject 'Microsoft.CognitiveServices/accounts/projects@2026-07-01' = {
  parent: foundry
  name: projectName
  location: location
  tags: tags
  identity: { type: 'SystemAssigned' }
  properties: {}
}

// The account accepts one change at a time, so the deployment waits for the project.
resource modelDeployment 'Microsoft.CognitiveServices/accounts/deployments@2026-07-01' = {
  parent: foundry
  name: modelDeploymentName
  sku: {
    name: 'GlobalStandard'
    capacity: modelCapacity
  }
  properties: {
    model: {
      format: 'OpenAI'
      name: modelName
      version: modelVersion
    }
    versionUpgradeOption: 'NoAutoUpgrade'
  }
  dependsOn: [ foundryProject ]
}

resource appFoundryUser 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(foundry.id, appIdentity.id, azureAiUserRoleId)
  scope: foundry
  properties: {
    roleDefinitionId: azureAiUserRoleId
    principalId: appIdentity.properties.principalId
    principalType: 'ServicePrincipal'
  }
}

resource developerFoundryUser 'Microsoft.Authorization/roleAssignments@2022-04-01' = if (!empty(principalId)) {
  name: guid(foundry.id, principalId, azureAiUserRoleId)
  scope: foundry
  properties: {
    roleDefinitionId: azureAiUserRoleId
    principalId: principalId
    principalType: 'User'
  }
}

resource appMetricsPublisher 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(appInsights.id, appIdentity.id, metricsPublisherRoleId)
  scope: appInsights
  properties: {
    roleDefinitionId: metricsPublisherRoleId
    principalId: appIdentity.properties.principalId
    principalType: 'ServicePrincipal'
  }
}

// -------------------------------------------------------------- Container Apps

resource environment 'Microsoft.App/managedEnvironments@2026-01-01' = {
  name: '${abbrs.environment}${resourceToken}'
  location: location
  tags: tags
  properties: {
    // Keyless logging: Azure Monitor plus the diagnostic setting below, instead of a shared key.
    appLogsConfiguration: { destination: 'azure-monitor' }
    workloadProfiles: [
      { name: 'Consumption', workloadProfileType: 'Consumption' }
    ]
  }
}

resource environmentLogs 'Microsoft.Insights/diagnosticSettings@2021-05-01-preview' = {
  name: 'logs-to-workspace'
  scope: environment
  properties: {
    workspaceId: workspace.id
    logAnalyticsDestinationType: 'Dedicated'
    logs: [
      { category: 'ContainerAppConsoleLogs', enabled: true }
      { category: 'ContainerAppSystemLogs', enabled: true }
    ]
  }
}

resource web 'Microsoft.App/containerApps@2026-01-01' = {
  name: '${abbrs.app}matchsense-${resourceToken}'
  location: location
  tags: union(tags, { 'azd-service-name': 'web' })
  identity: {
    type: 'UserAssigned'
    userAssignedIdentities: { '${appIdentity.id}': {} }
  }
  properties: {
    managedEnvironmentId: environment.id
    workloadProfileName: 'Consumption'
    configuration: {
      activeRevisionsMode: 'Single'
      ingress: {
        external: true
        targetPort: 8000
        // 'auto' keeps HTTP/1.1 upgrades working, which the live websocket stream needs.
        transport: 'auto'
        allowInsecure: false
      }
      registries: [
        { server: registry.properties.loginServer, identity: appIdentity.id }
      ]
    }
    template: {
      scale: {
        // One warm replica so judges never wait for a cold start; scale out on concurrent streams.
        minReplicas: 1
        maxReplicas: 3
        rules: [
          { name: 'http-concurrency', http: { metadata: { concurrentRequests: '30' } } }
        ]
      }
      containers: [
        {
          name: 'web'
          // Placeholder until `azd deploy` pushes the MatchSense image.
          image: 'mcr.microsoft.com/azuredocs/containerapps-helloworld:latest'
          resources: { cpu: json('1.0'), memory: '2Gi' }
          env: [
            { name: 'MATCHSENSE_NARRATION', value: 'auto' }
            { name: 'FOUNDRY_PROJECT_ENDPOINT', value: 'https://${foundryName}.services.ai.azure.com/api/projects/${projectName}' }
            { name: 'FOUNDRY_MODEL_DEPLOYMENT', value: modelDeploymentName }
            { name: 'FOUNDRY_USE_AGENTS', value: useFoundryAgents ? '1' : '0' }
            { name: 'AZURE_CLIENT_ID', value: appIdentity.properties.clientId }
            { name: 'APPLICATIONINSIGHTS_CONNECTION_STRING', value: appInsights.properties.ConnectionString }
            { name: 'OTEL_SERVICE_NAME', value: 'matchsense-web' }
          ]
          probes: [
            { type: 'Liveness', httpGet: { path: '/api/health', port: 8000 }, periodSeconds: 15 }
            { type: 'Readiness', httpGet: { path: '/api/health', port: 8000 }, periodSeconds: 10 }
          ]
        }
      ]
    }
  }
  dependsOn: [ acrPull, appFoundryUser, appMetricsPublisher, modelDeployment ]
}

output registryLoginServer string = registry.properties.loginServer
output registryName string = registry.name
output foundryProjectEndpoint string = 'https://${foundryName}.services.ai.azure.com/api/projects/${projectName}'
output modelDeploymentName string = modelDeployment.name
output webUrl string = 'https://${web.properties.configuration.ingress.fqdn}'
