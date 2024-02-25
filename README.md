# Asset Management Toolkit
The Asset Management Toolkit is Python library designed to facilitate asset management tasks using Python and artificial intelligence techniques. This toolkit offers a range of functions and utilities to streamline various aspects of asset management processes.
For Example (Socks analysis, CPPI, Efficient frontier , Different Syles of Allocations, Sytle analysis, Sentiment News Analysis)

## Features
- **Python-Based Functionality:** Leverage the power of Python programming language for asset management tasks.

- **AI Integration:** Incorporates artificial intelligence techniques for enhanced asset analysis and management.

- **Documentation:** Well-documented functions and modules for seamless usage.

## Usage

```javascript
import Asset_Management as am
```
### Examples
#### 1- Summary Stats
```javascript
am.summary_stats(r = Data , riskfree_rate=.16)
``` 
![st](https://github.com/AbdrahmanRaafat/Asset-Management/assets/74067572/57a3adec-c45a-49b2-a520-236575e3e8d5)

#### 2- Efficient Frontier
```javascript
target_risk={'EAST':.2, "SWDY":.1, "ABUK": .25 ,'HRHO':.05,
'COMI':.15, 'TMGH':.05 , 'AMOC':.05, 'ETEL':.15} #make sure that the sum of them = 1 

am.interactive_ef(n_points=20, er= er, cov = cov, show_gmv= True, show_cml= True , show_ew= True
, show_erc = True, show_trc = True, target_risk = target_risk, riskfree_rate= .16 )
 ```
![newplot ](https://github.com/AbdrahmanRaafat/Asset-Management/assets/74067572/015d1ef6-5c18-4dea-a8ae-8ada3899980f)

 #### 3- CPPI
```javascript
 cppi = am.run_cppi(risky_r = ra  #risky_asset 
                   ,riskfree_rate= rf #safe_asset
                   ,floor = .8 #80%
                   , m = 3 #asset_muliplier
                   , start = 1000 #Start Value
                   , drawdown=  .2 #maximum drawdown 
                  )
  ```
![1](https://github.com/AbdrahmanRaafat/Asset-Management/assets/74067572/783c32e7-660d-415c-9e66-9ec8c2c7e0e1)
![2](https://github.com/AbdrahmanRaafat/Asset-Management/assets/74067572/8c79fa5a-36d9-4d28-a204-88e56d78aeb3)

## Contact
For questions, suggestions, or support regarding the Asset Management Toolkit, feel free to contact me at rahman.raafat@yahoo.com or 
[![linkedin](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/abdelrahman-r-22b96211a/)
.
