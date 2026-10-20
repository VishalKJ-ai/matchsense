// MatchSense on Azure: one resource group with Container Apps, Microsoft Foundry and monitoring.
// Deploy with the Azure Developer CLI: `azd up` (see README, "Deploy to Azure").
targetScope = 'subscription'

@minLength(1)
@maxLength(40)
@description('Name of the azd environment; used to name the resource group and resources.')
param environmentName string

@minLength(1)
@description('Region for every resource. Pick one where your Foundry model is available.')
param location string

@description('Object ID of the person deploying (set by azd), granted Foundry access for local development.')
param principalId string = ''

@description('Foundry model to deploy for the narration crew.')
param modelName string = 'gpt-5-mini'

@description('Version of that model.')
param modelVersion string = '2025-08-07'

@description('Model capacity in thousands of tokens per minute.')
param modelCapacity int = 50

@description('Register each crew role as a versioned Foundry prompt agent (preview). Direct calls are used otherwise.')
param useFoundryAgents bool = true

@description('GitHub repository whose main branch may deploy through OIDC, as in its OIDC subject: owner/name, or owner@ownerId/name@repoId when GitHub uses immutable IDs. Leave empty to skip.')
param githubRepository string = ''

@description('Whether the web app already exists (set by azd), so provisioning keeps its current image.')
param webExists bool = false

@description('Email for budget alerts. Leave empty to skip the budget.')
param budgetContactEmail string = ''

@description('Monthly budget in USD.')
param budgetAmount int = 50

@description('First day of the budget, yyyy-MM-01. Defaults to the current month.')
param budgetStartDate string = '${utcNow('yyyy-MM')}-01'

var tags = { 'azd-env-name': environmentName, project: 'matchsense' }

resource rg 'Microsoft.Resources/resourceGroups@2024-03-01' = {
  name: 'rg-${environmentName}'
  location: location
  tags: tags
}

module resources 'resources.bicep' = {
  name: 'resources'
  scope: rg
  params: {
    location: location
    tags: tags
    resourceToken: resourceToken
    principalId: principalId
    modelName: modelName
    modelVersion: modelVersion
    modelCapacity: modelCapacity
    useFoundryAgents: useFoundryAgents
    githubRepository: githubRepository
    webImage: webImage.outputs.image
  }
}

var resourceToken = toLower(uniqueString(subscription().id, environmentName, location))

// Provisioning re-applies the app template; reuse the deployed image instead of the placeholder.
module webImage 'fetch-image.bicep' = {
  name: 'web-image'
  scope: rg
  params: {
    exists: webExists
    name: 'ca-matchsense-${resourceToken}'
  }
}

module budget 'budget.bicep' = if (!empty(budgetContactEmail)) {
  name: 'budget'
  scope: rg
  params: {
    name: 'matchsense-${environmentName}-monthly'
    amount: budgetAmount
    contactEmail: budgetContactEmail
    startDate: budgetStartDate
  }
}

output AZURE_LOCATION string = location
output AZURE_RESOURCE_GROUP string = rg.name
output AZURE_CONTAINER_REGISTRY_ENDPOINT string = resources.outputs.registryLoginServer
output AZURE_CONTAINER_REGISTRY_NAME string = resources.outputs.registryName
output FOUNDRY_PROJECT_ENDPOINT string = resources.outputs.foundryProjectEndpoint
output FOUNDRY_MODEL_DEPLOYMENT string = resources.outputs.modelDeploymentName
output WEB_URL string = resources.outputs.webUrl
output AZURE_CONTAINER_APP_NAME string = resources.outputs.webAppName
output AZURE_DEPLOY_CLIENT_ID string = resources.outputs.deployClientId
output AZURE_TENANT_ID string = tenant().tenantId
