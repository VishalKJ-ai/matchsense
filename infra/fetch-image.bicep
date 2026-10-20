// The image a container app currently runs, or '' when the app does not exist yet.
targetScope = 'resourceGroup'

param exists bool
param name string

resource app 'Microsoft.App/containerApps@2026-01-01' existing = if (exists) {
  name: name
}

output image string = exists ? app!.properties.template.containers[0].image : ''
