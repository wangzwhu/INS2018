﻿## Guide Line for INS 2018 task
* [https://www-nlpir.nist.gov/projects/tv2018/Tasks/instance-search/](https://www-nlpir.nist.gov/projects/tv2018/Tasks/instance-search/)


## Schedule for INS 2018 task
* June 4 : Submission webpage open (trial submissions highly recommended) 
* June 17: Run submissions due at NIST by NOON Washington, DC time

## Topics for INS 2018
> Automatic runs (30) topics : **ins.auto.topics.2018.xml**  
> Interactive runs (21) topics : **ins.inter.topics.2018.xml**  
> Master location topics file (stating the different locations with their corresponding images and shot examples) : **ins.location.topics**  
> Location image examples folder : **location.example.images/**  
> Location shot examples folder : **location.example.shots/**  
> Person image examples folder : **tv18.person.example.images/**  
> Person shot examples folder : **tv18.person.example.shots/**  

## Person ID
|ID|Name| |
|---|---|---|
| 1 |chelsea| ![](./sample/p-1.jpg =100x100)|
| 2 |darrin|![](./sample/p-2.jpg =100x100)|
| 3 |garry|![](./sample/p-3.jpg =100x100)|
| 4 |heather|![](./sample/p-4.jpg =100x100)|
| 5 |jack|![](./sample/p-5.jpg =100x100)|
| 6 |jane|![](./sample/p-6.jpg =100x100)|
| 7 |max|![](./sample/p-7.jpg =100x100)|
| 8 |minty|![](./sample/p-8.jpg =100x100)|
| 9 |mo|![](./sample/p-9.jpg =100x100)|
| 10 |zainab|![](./sample/p-10.jpg =100x100)|

* face_dict_path = `/net/per920a/export/das14a/satoh-lab/wangz/ins2018/face_dict`

## Scene ID
| ID   | Name | |
|-|-|-|
| 1 |cafe1| ![](./sample/cafe1.bmp =100x100)|
| 2 |cafe2| ![](./sample/cafe2.bmp =100x100)|
| 3 |foyer| ![](./sample/foyer.bmp =100x100)|
| 4 |kitchen1| ![](./sample/kitchen1.bmp =100x100)|
| 5 |kitchen2| ![](./sample/kitchen2.bmp =100x100)|
| 6 |laun| ![](./sample/laun.bmp =100x100)|
| 7 |LR1| ![](./sample/LR1.bmp =100x100)|
| 8 |LR2| ![](./sample/LR2.bmp =100x100)|
| 9 |market| ![](./sample/market.bmp =100x100)|
| 10 |pub| ![](./sample/pub.bmp =100x100)|

## Topics
| Topic| P + S        |  P_id + S_id |
|-|-|-|
| 9219 |  Jane+Cafe2  |      **6+2** | 
| 9220 |  Jane+Pub    |      **6+10**| 
| 9221 | Jane+Market  |    **6+9**   |
| 9222 | Chelsea+Cafe2|    **1+2**   | 
| 9223 | Chelsea+Pub  |    **1+10**  | 
| 9224 | Chelsea+Market|   **1+9**   |
| 9225 | Minty+Cafe2   |   **8+2**   |
| 9226 | Minty+Pub    |    **8+10**  |
| 9227 | Minty+Market |    **8+9**   |
| 9228 | Garry+Cafe2  |     **3+2**  |
| 9229 | Garry+Pub    |     **3+10** |
| 9230 | Garry+Laundrette| **3+6**   |
| 9231 | Mo+Cafe2     |    **9+2**   |
| 9232 | Mo+Pub       |    **9+10**  |
| 9233 | Mo+Laundrette|    **9+6**   |
| 9234 | Darrin+Cafe2 |    **2+2**   |
| 9235 | Darrin+Pub   |    **2+10**  |
| 9236 | Darrin+Laundrette|**2+6**   |
| 9237 | Zainab+Cafe2 |    **10+2**  |
| 9238 | Zainab+Laundrette|**10+6**  |
| 9239 | Zainab+Market  |  **10+9**  |
| 9240 | Heather+Cafe2  |  **4+2**   |
| 9241 | Heather+Laundrette| **4+6** |
| 9242 | Heather+Market |  **4+9**   |
| 9243 | Jack+Pub       |  **5+10**  |
| 9244 | Jack+Laundrette|  **5+6**   |
| 9245 | Jack+Market    |  **5+9**   |
| 9246 | Max+Cafe2      |  **7+2**   |
| 9247 | Max+Laundrette |  **7+6**   |
| 9248 | Max+Market     |  **7+9**   |

## Reference
* Facenet: [https://github.com/davidsandberg/facenet](https://github.com/davidsandberg/facenet)
* Deep-Retrieval: [https://github.com/figitaki/deep-retrieval](https://github.com/figitaki/deep-retrieval)
