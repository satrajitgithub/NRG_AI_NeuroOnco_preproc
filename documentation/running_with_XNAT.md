# Running on XNAT (preferred)
### 1. Installation and Requirements
!!!!!!!!! XNAT version
#### 1.1. XNAT setup
Please follow these steps to set up XNAT, import data, and enable container service:
1. [Set up XNAT](https://wiki.xnat.org/documentation/getting-started-with-xnat/xnat-installation-guide)
2. [Create a project](https://wiki.xnat.org/documentation/how-to-use-xnat/creating-and-managing-projects)
3. [Import data](https://wiki.xnat.org/documentation/how-to-use-xnat/image-session-upload-methods-in-xnat)
4. [Container service administration](https://wiki.xnat.org/container-service/container-service-administration-122978855.html)
#### 1.2. Pulling and enabling containers
Once container service is set up, [pull the following docker images](https://wiki.xnat.org/container-service/pulling-a-container-image-126156950.html) to XNAT:
* `satrajit2012/nrg_ai_neuroonco_preproc:v0`
* `satrajit2012/nrg_ai_neuroonco_segment:v0`
 
In order to launch containers, the corresponding container commands need to be first enabled at the site level (by a site admin) and then at the project-level (this can be done by a project owner). For details on how to enable commands, please check out the following links:
1. [Enabling commands at site-level](https://wiki.xnat.org/container-service/enabling-commands-and-setting-site-wide-defaults-126156956.html)
2. [Enabling commands at project-level](https://wiki.xnat.org/container-service/enable-a-command-in-your-project-122978909.html)

Now with everything set up, we are ready to run the containers.
### 2. Running containers
There are various ways we can run the containers:
* manually run multiple containers on single session
* manually run single container on multiple sessions (using XNAT's batch-mode feature)
* automatically run multiple containers on single session (using XNAT's command orchestration feature)
* automatically run multiple containers on multiple sessions (by combining XNAT's command orchestration with batch-launch feature)

In this section I describe the above 4 modes of running the containers.
#### 2.1. Manually run multiple containers on single session
In this step, you can launch containers through the "Run Containers" option from the XNAT UI. Note that, as these containers are defined at a session-level context (i.e. they run on each session independently, not on a subject or project level), the "Run Containers" menu should only appear within a session, as following:
![](figures/launch_session_level.png)
Inside the "Run Containers" menu are all the different command wrappers that can be used to launch the different workflows on the session. <br />
For more details on this, please refer to: [Launching Containers from Commands](https://wiki.xnat.org/container-service/launching-containers-from-commands-122978910.html)

Note that in this step you need to launch each command manually one after other. For example, launch `1-Scan-type Classifier` and wait till its done. Then you can launch `2-Registration` and so on. This can be useful if you need to launch only specific commands of the pipeline on a particular session. However, doing this for every session in a big dataset can be quite tedious. We can address that using XNAT's batch-launch and command orchestration as described later.

#### 2.2. Manually run single container on multiple sessions
In a dataset containing a large number of sessions, it is often necessary to launch a particular command on multiple sessions. This is where [XNAT's Batch Launch Plugin](https://wiki.xnat.org/xnat-tools/batch-launch-plugin) can be particularly useful. Once the Batch Launch Plugin is [installed](https://wiki.xnat.org/xnat-tools/batch-launch-plugin#:~:text=Installing%20the%20Batch%20Launch%20Plugin), you can follow the steps detailed here to bulk-launch a command on session-level: [Using the Batch Launch Plugin with the Container Service](https://wiki.xnat.org/xnat-tools/batch-launch-plugin/using-the-batch-launch-plugin-with-the-container-service). <br />
![](figures/batch-launch.gif)
Once bulk processing has finished, the Processing Dashboard lists the outcome (Complete/Failed/Ready) as a sortable column which can be used to additionally inspect the results.
#### 2.3. Automatically run multiple containers on single session
In this step, we will take advantage of XNAT's [Command Orchestration](https://wiki.xnat.org/container-service/set-up-command-orchestration-130515311.html)
#### 2.4. Automatically run multiple containers on multiple sessions




