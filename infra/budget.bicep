// A monthly cost budget with email alerts at 50%, 80% and 100%, and a forecast alert.
targetScope = 'resourceGroup'

param name string
param amount int
param contactEmail string
param startDate string

resource budget 'Microsoft.Consumption/budgets@2023-11-01' = {
  name: name
  properties: {
    category: 'Cost'
    amount: amount
    timeGrain: 'Monthly'
    timePeriod: { startDate: startDate }
    notifications: {
      Actual50: { enabled: true, operator: 'GreaterThan', threshold: 50, contactEmails: [ contactEmail ], thresholdType: 'Actual' }
      Actual80: { enabled: true, operator: 'GreaterThan', threshold: 80, contactEmails: [ contactEmail ], thresholdType: 'Actual' }
      Actual100: { enabled: true, operator: 'GreaterThan', threshold: 100, contactEmails: [ contactEmail ], thresholdType: 'Actual' }
      Forecast100: { enabled: true, operator: 'GreaterThan', threshold: 100, contactEmails: [ contactEmail ], thresholdType: 'Forecasted' }
    }
  }
}
