# Nano-cap Value

An automated approach to identifying undervalued nano-cap stocks using SEC XBRL filings data.

## Core Thesis

The fundamental value of a business consists of two components:

```math
\text{Total Value} = \text{Net Asset Value} + \text{Present Value of Future Cash Flows}
```

Investors pay a premium for liquidity. Therefore, illiquid stocks may have higher value. Also, small market cap stocks may not be worth it for larger players, so there may be more inefficiencies left unexploited.

## 1. Asset Value Calculation

### Components
- Cash and Cash Equivalents (100% of value)
  - Bank deposits
  - Money market funds
  - Short-term securities
  - Marketable securities

- Accounts Receivable (discounted value)
- Inventory (discounted value)
- Fixed Assets (discounted value)
  - Property, plant, and equipment
  - Land
  - Buildings
  - Machinery

### Net Asset Value Formula

```math
NAV = (Cash) + (Receivables \times \alpha) + (Inventory \times \beta) + (PP\&E \times \gamma) - Total\space Liabilities
```

### Liabilities
1. Current Liabilities
   - Accounts payable
   - Short-term debt
   - Accrued expenses
   - Current portion of long-term debt
   - Income taxes payable
   - Customer deposits

2. Long-term Liabilities
   - Long-term debt
   - Capital lease obligations
   - Pension obligations
   - Deferred tax liabilities

## 2. Operating Value Calculation

### Free Cash Flow Calculation

```math
Operating\space Cash\space Flow = Revenue - Operating\space Expenses \pm \Delta Working\space Capital
```

```math
FCF = Operating\space Cash\space Flow - Average\space Historical\space CapEx
```

### Present Value Calculation
For a diversified portfolio of micro-cap stocks:

```math
Present\space Value = \sum_{t=1}^{n} \frac{FCF_t}{(1 + r)^t}
```

Where:
- $FCF_t$ = Free Cash Flow in year t
- $r$ = Opportunity cost rate (e.g., S&P 500 expected return)
- $t$ = Year number

## Implementation

### SEC Filing Data Collection

#### Some XBRL Tags (possible)
```
us-gaap:CashAndCashEquivalentsAtCarryingValue
us-gaap:AccountsReceivableNetCurrent
us-gaap:InventoryNet
us-gaap:PropertyPlantAndEquipmentNet
us-gaap:AccountsPayableCurrent
us-gaap:LongTermDebtNoncurrent
us-gaap:Revenues
us-gaap:CostOfRevenue
us-gaap:OperatingExpenses
us-gaap:CapitalExpendituresIncurredButNotYetPaid
us-gaap:PaymentsToAcquirePropertyPlantAndEquipment
us-gaap:OperatingLeaseLiabilityCurrent
us-gaap:ContractWithCustomerLiabilityCurrent
us-gaap:AccruedLiabilitiesCurrent
us-gaap:OperatingLeaseLiabilityNoncurrent
us-gaap:DeferredTaxLiabilitiesNoncurrent
us-gaap:PensionAndOtherPostretirementBenefitPlans
us-gaap:WeightedAverageNumberOfSharesOutstandingBasic
us-gaap:CommonStockParOrStatedValuePerShare
us-gaap:CommitmentsAndContingencies
us-gaap:GuaranteeObligations
us-gaap:GrossProfit
us-gaap:NetIncomeLoss
us-gaap:EarningsPerShareBasic
us-gaap:EarningsPerShareDiluted
us-gaap:InterestExpense
us-gaap:IncomeTaxExpenseBenefit
us-gaap:TotalLiabilities
us-gaap:TotalAssets
us-gaap:DepreciationAmortizationAndAccretion
```

#### Value Ratio Calculation

```math
\text{Value Ratio} = \frac{\text{Total Value}}{\text{Market Cap}} = \frac{NAV + \text{Present Value of FCF}}{\text{Market Cap}}
```

Where:
```math
NAV = (Cash) + (Receivables \times \alpha) + (Inventory \times \beta) + (PP\&E \times \gamma) - Total\space Liabilities
```

```math
\text{Present Value of FCF} = \sum_{t=1}^{n} \frac{FCF_t}{(1 + r)^t}
```

#### Primary Screens

1. Market Cap Screen
   - Below $50M market cap

2. Value Screen
   - Calculate Value Ratio (average over some time period)
   - Flag companies trading below some multiple of Total Value

Take the top % of best valued stocks, use for portfolio construction.

### Portfolio Construction

To construct an optimal portfolio, we need to generate the efficient frontier - the set of portfolios that give the highest expected return for each level of risk. First, we define our inputs:

Let $w$ be our vector of portfolio weights (what we're solving for). We need two key quantities:

```math
\mu_i = \text{Value Ratio}_i \quad \text{(expected return for stock i)}
```

```math
\Sigma_{ij} = \frac{1}{T-1}\sum_{t=1}^T (R_{it} - \bar{R}_i)(R_{jt} - \bar{R}_j) \quad \text{(covariance between stocks i and j)}
```

where $R_{it}$ is the realized return of asset $i$ at time $t$, $T$ is the number of historical observations, and $\bar{R}_i$ is the mean historical return of asset $i$.

To generate the efficient frontier, we solve:

```math
\min_w w^T\Sigma w \quad \text{(minimize portfolio variance)}
```

subject to:
```math
w^T\mu = \mu_{\text{target}} \quad \text{(achieve target return)}
```
```math
\sum_{i=1}^n w_i = 1 \quad \text{(fully invested)}
```
```math
0 \leq w_i \leq 0.20 \quad \text{(position limits)}
```

By solving this for different values of $\mu_{\text{target}}$, we generate the efficient frontier. Each point on the frontier represents a portfolio with minimum variance for its target return level.

To choose our position on the frontier:

To choose the position on the frontier, choose your desired annual return $y$ above the opportunity cost rate $r$. Given an expected time to fair value of $b$ years, set:

```math
\mu_{\text{target}} = (1 + y)^b
```

For example:
- For 20% annual return with 3-year convergence: $\mu_{\text{target}}$ = (1 + 0.20)^3

The higher the target return $y$ or longer the convergence time $b$, the higher the required $\mu_{\text{target}}$ and corresponding portfolio variance.
