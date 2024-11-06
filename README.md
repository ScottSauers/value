# Nano-Cap Value

An automated approach to identifying undervalued nano-cap stocks using SEC XBRL filings data.

## Core Thesis

The fundamental value of a business consists of two components:

```math
\text{Total Value} = \text{Net Asset Value} + \text{Present Value of Future Cash Flows}
```

Investors pay a premium for liquidity. Therefore, illiquid stocks may have higher value. Also, small market cap stocks may not be worth it for larger players, so there may be more inefficiencies left unexploited. As the market recognizes the true value of the company, the stock trend towards fair value.

## Usage
- Install dependencies.
- Create an .env file with your API keys.
- Run screen.py to get small cap stocks.
- Run all_fundamentals.py to get fundamentals.

## Asset Value Calculation

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

### Asset Discounts

Accounts Receivable (α):
```math
α = \frac{1}{1 + \frac{\text{Average Net Accounts Receivable}}{\text{Average Annual Revenue}}}
```
This measures collection efficiency. A company collecting receivables in 3 months (AR = Revenue/4) gets α = 0.8, while one taking a year gets α = 0.5. Net Accounts Receivable already incorporates management's assessment of collectibility through bad debt allowances.

Inventory (β):
```math
β = \frac{1}{1 + \frac{\text{Average Inventory}}{\text{Average Annual Cost of Goods Sold}}}
```
This captures inventory liquidity through turnover. Inventory turning 4 times yearly (Inventory = COGS/4) gets β = 0.8, while turning once yearly gets β = 0.5. Faster-moving inventory commands higher recovery rates in liquidation.

Property, Plant & Equipment (γ):
```math
γ = \frac{1}{1 + \frac{\text{Average Net PP\&E}}{\text{Average Annual Revenue}}}
```
This measures asset productivity - more revenue per dollar of assets indicates more marketable/valuable assets. PP&E generating its value in revenue annually (PP&E = Revenue) gets γ = 0.5, while generating 4x its value gets γ = 0.8.

All three parameters must:
- Use only universally reported financial data
- Generate dimensionless ratios between 0 and 1

These might mix point-in-time with averages.

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

## Operating Value Calculation

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

#### Value Ratio Calculation

```math
\text{Value Ratio} = \frac{\text{Total Value}}{\text{Market Cap}} = \frac{NAV + \text{Present Value of FCF} + \text{Present Value of Contracted Revenue}}{\text{Market Cap}}
```

Where:
```math
NAV = (Cash) + (Receivables \times \alpha) + (Inventory \times \beta) + ((PP\&E + Operating\space Lease\space Assets) \times \gamma) - Total\space Liabilities
```

```math
\text{Present Value of FCF} = \sum_{t=1}^{n} \frac{FCF_t}{(1 + r)^t}
```

#### Value Ratio Calculation

```math
\text{Value Ratio} = \frac{\text{Total Value}}{\text{Market Cap}} = \frac{NAV + \text{Present Value of FCF}}{\text{Market Cap}}
```

Where:
```math
NAV = (Cash) + (Receivables \times \alpha) + (Inventory \times \beta) + ((PP\&E + Operating\space Lease\space Assets) \times \gamma) - Total\space Liabilities
```

```math
\text{Present Value of FCF} = \sum_{t=1}^{n} \frac{FCF_t}{(1 + r)^t}
```

### Value Ratio Calculation Methodology

**Core Formula**  
```math
\text{Value Ratio} = \frac{\text{Net Asset Value} + \text{Present Value of Future Cash Flows}}{\text{Market Cap}}
```

**Required Data Components**

**Absolute Requirements**  
Must have ALL of:
- Market cap (calculated from price × `CommonStockSharesOutstanding`)
- `CashAndCashEquivalentsAtCarryingValue`
- Liabilities or complete liability construction (see Liabilities section)
- Revenues or `RevenueFromContractWithCustomerExcludingAssessedTax`
- At least one of: `AccountsReceivableNetCurrent`, `PropertyPlantAndEquipmentNet`, or `InventoryNet`

**Periods Required**

**Point-in-Time Measures (Most Recent Quarter)**:
- `CommonStockSharesOutstanding`
- `CashAndCashEquivalentsAtCarryingValue`
- `AvailableForSaleSecurities`
- `AccountsReceivableNetCurrent`
- `InventoryNet`
- `PropertyPlantAndEquipmentNet`
- `OperatingLeaseRightOfUseAsset` (if available)
- `IntangibleAssetsNetExcludingGoodwill`
- All liability components
- All asset components

**Trailing Twelve Month (TTM) Measures**:
- `Revenues` or `RevenueFromContractWithCustomerExcludingAssessedTax`
- `CostOfGoodsAndServicesSold` or `CostOfRevenue`
- `NetCashProvidedByUsedInOperatingActivities`
- `PaymentsToAcquirePropertyPlantAndEquipment`
- `CapitalExpendituresIncurredButNotYetPaid` (if available)
- `PaymentsToAcquireBusinessesNetOfCashAcquired` (if available)
- `OperatingIncomeLoss`
- `DepreciationDepletionAndAmortization`

**Component Calculations**

1. **Cash and Marketable Securities**  
   - Most Recent Quarter: Primary: `CashAndCashEquivalentsAtCarryingValue`
   - Add if available: `AvailableForSaleSecurities`

2. **Accounts Receivable**  
   - Most Recent Quarter Amount: `AccountsReceivableNetCurrent`
   - Discount Rate α calculation:
     ```math
     \alpha = \frac{1}{1 + \frac{\text{AccountsReceivableNetCurrent}}{\text{Revenue}_{\text{TTM}}}}
     ```

3. **Inventory**  
   - Most Recent Quarter Amount: `InventoryNet`
   - Discount Rate β calculation:
     ```math
     \beta = \frac{1}{1 + \frac{\text{InventoryNet}}{\text{Cost}_{\text{TTM}}}}
     ```

4. **Property, Plant & Equipment**  
   - Most Recent Quarter Amount: `PropertyPlantAndEquipmentNet` + `OperatingLeaseRightOfUseAsset` (if available)
   - Discount Rate γ calculation:
     ```math
     \gamma = \frac{1}{1 + \frac{\text{PropertyPlantAndEquipmentNet} + \text{OperatingLeaseRightOfUseAsset}}{\text{Revenue}_{\text{TTM}}}}
     ```
  - Alternative (if primary missing):
      - `PropertyPlantAndEquipmentNet` = `PropertyPlantAndEquipmentGross` - `AccumulatedDepreciationDepletionAndAmortizationPropertyPlantAndEquipment`

5. **Liabilities**  
   - Most Recent Quarter: Primary Method: `Liabilities`
   - Alternative Construction (sum of) (if primary missing):
     - `LiabilitiesCurrent`
     - `LongTermDebtNoncurrent`
     - `OperatingLeaseLiabilityNoncurrent`
     - `DeferredTaxLiabilitiesNoncurrent`
     - `OtherLiabilitiesNoncurrent`

6. **Operating Asset Value**  
   - Most Recent Quarter: `IntangibleAssetsNetExcludingGoodwill` (if available)
   - Operating Asset multiplier based on industry classification

**Free Cash Flow (FCF) Calculation**

**Primary Method (TTM Basis)**  
- Operating Cash Flow: Sum last four quarters of `NetCashProvidedByUsedInOperatingActivities`
- Capital Expenditure: Sum last four quarters of (`PaymentsToAcquirePropertyPlantAndEquipment` + `CapitalExpendituresIncurredButNotYetPaid` (if available) - `PaymentsToAcquireBusinessesNetOfCashAcquired` (if available))
```math
\text{FCF} = \text{Operating Cash Flow} - \text{Capital Expenditure}
```

**Alternative Method (TTM Basis)**  
If operating cash flow unavailable:
```math
\text{FCF} = (\text{OperatingIncomeLoss} + \text{DepreciationDepletionAndAmortization}) - (\text{IncreaseDecreaseInAccountsReceivable} + \text{IncreaseDecreaseInPrepaidDeferredExpenseAndOtherAssets}) - \text{PaymentsToAcquirePropertyPlantAndEquipment}
```

**TTM Construction Rules**  
For any flow measure:
- Primary: Sum most recent four quarters
- Missing Quarter Handling:
  - If one quarter missing between available quarters: Linear interpolation
  - If two non-adjacent quarters available: Scale to annual rate
  - If only one quarter available: Cannot calculate TTM
  - If quarters missing at end: Use older quarters if within 180 days

**Present Value Calculation**  
If current TTM FCF is positive:
```math
\text{Present Value} = \frac{\text{FCF}_{\text{TTM}}}{0.10}
```
If current TTM FCF is negative:
- Examine each of previous four TTM periods
- Use most recent positive FCF if available
- If no positive FCF in lookback period:
  ```math
  \text{Present Value} = 0
  ```

**Net Asset Value Assembly**  
```math
\text{NAV} = \text{Cash Components} + \text{Discounted Operating Assets} - \text{Total Liabilities}
```
where:
```math
\text{Cash Components} = \text{CashAndCashEquivalentsAtCarryingValue} + \text{AvailableForSaleSecurities}
```
```math
\text{Operating Assets} = (\text{AccountsReceivableNetCurrent} \times \alpha) + (\text{InventoryNet} \times \beta) + (\text{PropertyPlantAndEquipmentNet} + \text{OperatingLeaseRightOfUseAsset}) \times \gamma + \text{IntangibleAssetsNetExcludingGoodwill}
```

**Data Quality Requirements**  
- Point-in-time measures must be from most recent quarter end
- TTM calculations require a minimum of three quarters
- Most recent quarter must be within 180 days
- Market cap must use current shares outstanding
- Operating fundamentals (Revenue, FCF) must show consistency across periods

**Warning Conditions**  
Flag if:
- Less than three quarters available for TTM
- Most recent quarter older than 180 days
- Required components missing
- Significant period-over-period variance (>50%) in operating metrics
- Negative stockholder equity (`StockholdersEquity`)
- Operating cash flow and FCF have opposite signs
- Operating lease assets present without corresponding liabilities or vice versa

## Portfolio Construction

#### Primary Screens

1. Market Cap Screen
   - Below $50M market cap

2. Value Screen
   - Calculate Value Ratio (average over some time period)
   - Flag companies trading below some multiple of Total Value

Take the top % of best valued stocks, use for portfolio construction. Keep this percentage liberal since the portfolio will be optimized for more efficient return later.

### Portfolio Weights

To construct an optimal portfolio, we need to generate the efficient frontier - the set of portfolios that give the highest expected return for each level of risk. First, we define our inputs:

Let $w$ be our vector of portfolio weights (what we're solving for). We need two key quantities:

```math
\mu_i = \text{Value Ratio}_i \quad \text{(expected return for stock i)}
```

```math
\Sigma_{ij} = \frac{1}{T-1}\sum_{t=1}^T (R_{it} - \bar{R}_i)(R_{jt} - \bar{R}_j) \quad \text{(covariance between stocks i and j)}
```

where $R_{it}$ is the realized return of asset $i$ at time $t$, $T$ is the number of historical observations, and $\bar{R}_i$ is the mean historical return of asset $i$.

Then, we can apply [Ollila-Raninen shrinkage](https://arxiv.org/pdf/1808.10188) (Ell3-RSCM) to the covariance matrix. View the proper code for it here: http://users.spa.aalto.fi/esollila/regscm/.

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

### Exit Strategy
Re-calculate the optimal portfolio weekly. Buy and sell the stocks with the largest dollar value changes until within a small percentage from optimality.
