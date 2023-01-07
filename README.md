# srf-presenter-detector
Given an SRF Tagesschau video file, detect text shown in overlays.
Its not just to detect the presenter but its too late to rename everything now.

![tagesschau](https://user-images.githubusercontent.com/24475986/211169413-5b305461-60d0-468f-97ef-3ebc62b767ca.PNG)

The /misc directory contains scripts used during the project, including the fetch_tagesschau_mp4.py which pulls a tagesschau show and splits it into frames. Used to prepare training data for the image segmentation model.

To build the application as a container, pull the repo and enter the /container directory. Build was done using buildah/podman but docker will work as well:
```
buildah bud -t srf-presenter-detector -f Dockerfile --layers .  
```

Run the application as follow:
```
podman run --rm -p 8080:8080 srf-presenter-detector
```
