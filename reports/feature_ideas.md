## Business-Motivated Feature Ideas
## Monthly burden 
loan_amount / loan_duration

## young borrower flag
age < 25

## High loan
loan_amount > median

## Long Employment Flag
employment_duration > threshold




### Debt Burden Proxy

Hypothesis:
Borrowers with higher repayment obligations relative to their financial capacity may be more likely to default.

Potential Features:
- monthly_burden = loan_amount / loan_duration
- burden_x_installment_rate = monthly_burden * installment_rate

Motivation:
Inspired by debt-to-income ratio concepts commonly used in credit risk assessment.
