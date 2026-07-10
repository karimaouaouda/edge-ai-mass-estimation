- the class distribution on the methodology. so make sure to validate it before rendering.


- the instructor name must not be in the cover page.


- the database used is not mysql, it's postgres sql, postgres sql is power full and designed for complex queries, and it is more suitable for the system, so make sure to change it in the thesis.


- in the use cases diagram it is so basic like the cases are pipeline, so make sure to distinct the cases.
    - currently the use cases are like that:
    ```
    actor -------> use case 1 ------> use case 2 ------> use case 3
    ```
    - instead of it, i need it like this : 
    ```
    actor -------> use case 1
          -------> use case 2
          -------> use case 3
    ```
    so the reader knows are distinct use cases, and not just a pipeline.


- jetson nano used is with 4gb memory with shared memory Nvidia GPU

- camera used is 1080p webcam, this is can affect results as it's not for indistrial uses.



- future work and improvements section additional points to add if aren't added:
    - to improve results it's good to
        - pretrain the model in a noisy environment (aquatrash + taco) then fine_tune on regular envs datasets (realwaste + trashnet) with applying lighting-augmentation and noise-augmentation to improve the model generalization and robustness, then fine tune on custom dataset that taken from the experiment environment, this will improve the model performance and generalization.

        - use RGB-D and depth camera to improve the model performance, and to improve the model generalization and robustness. and get a more accurate results.



- I added a screen shots about inference pipeline from dashboard, so make sure to add them in the proper place in the thesis, put them in the section that explain the operation or the component that the image is related to.
- i added a visual screenshots about depth and seg, and mass results,  so make sure to add them in the proper place in the thesis, put them in the section that explain the operation or the component that the image is related to.