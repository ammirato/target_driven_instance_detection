##source /opt/ros/indigo/setup.bash
#
#
#bvision8=1;
#
#
#
#if (( bvision8 == 0)); then
#
##    echo bvisionserver1
#
#    ##bvisionserver1
#    export PATH=$PATH:/usr/local/cuda-6.5/bin
#    export LD_LIBRARY_PATH=:/usr/local/cuda-6.5/lib64
#    export PATH=/usr/local/cuda/bin:${PATH}
#
#    # Uncomment the first one to run fast-rcnn and uncomment second one to run segmentation. I dont think both should be uncommented at same time
#    export PYTHONPATH=/playpen2/poirson/fast-rcnn/caffe-fast-rcnn/python:${PYTHONPATH}
#    #export PYTHONPATH=/playpen2/poirson/caffe/python:${PYTHONPATH}
#
#    export PYTHONPATH=/usr/include/python2.7:/usr/lib/python2.7/dist-packages/numpy/core/include:${PYTHONPATH}
#    export PYTHONPATH=/usr/lib/python2.7/dist-packages:${PYTHONPATH}
#    export PYTHONPATH=/usr/local/lib/python2.7/dist-packages:/usr/lib/pymodules/python2.7:${PYTHONPATH}
#
#    export PYTHONPATH=/usr/local/lib/python2.7/dist-packages:${PYTHONPATH}
#
#
#    export PYTHONPATH=/usr/local/lib/python2.7/dist-packages:${PYTHONPATH}
#
#
#
#else
#
#
# #   echo bvision8
#
#    ####  bvision8
#    export PATH=/usr/local/cuda-7.5/bin:${PATH}
#    export LD_LIBRARY_PATH=/usr/local/cuda-7.5/lib64:${LD_LIBRARY_PATH}
#    export LD_LIBRARY_PATH=/playpen/poirson/anaconda/lib:${LD_LIBRARY_PATH}
#    #export PYTHONPATH=/home/poirson/visionserver/fast-rcnn/caffe-fast-rcnn/python:/usr/local/lib/python2.7/dist-packages:/home/poirson/selective_search_ijcv_with_python:/home/poirson/
#    #export PYTHONPATH=/home/poirson/visionserver/caffe/python:/usr/lib/python2.7/dist-packages
#
#    #export PATH=/usr/local/cuda/bin:${PATH}
#    #export PATH=${CUDA_HOME}/bin:${PATH} 
#    export PATH=$PATH:/usr/local/cuda/bin
#    #export LD_LIBRARY_PATH=:/usr/local/cuda/lib64:${LD_LIBARY_PATH}
#    export PATH=/home/poirson/visionserver/selective_search_ijcv_with_python:${PATH}
#    export PATH=/home/poirson/visionserver/rcnn-depth/eccv14-code/rgbdutils/imagestack/matlab:${PATH}
#    export PATH=/home/poirson/visionserver/rcnn-depth/eccv14-code/rgbdutils/imagestack/matlab/joint_bilateral_mex.mexa64:${PATH}
#
#
#    # added by Anaconda 2.3.0 installer
#    export PATH="/playpen/poirson/anaconda/bin:$PATH"
#
#    export PATH=/playpen/poirson/anaconda/include/python2.7/pyconfig.h:${PATH}
#    export PYTHONPATH=/playpen/poirson/anaconda/include/python2.7/pyconfig.h:${PYTHONPATH}
#    #export PYTHONPATH=/playpen/poirson/fast-rcnn/caffe-fast-rcnn/python:${PYTHONPATH}
#    export PATH=/usr/local/lib:${PATH}
#    export LD_LIBRARY_PATH=/usr/local/lib:${LD_LIBRARY_PATH}
#    export PATH="$PATH:$HOME/bin"
#    export LD_LIBRARY_PATH=/usr/local/cuda-7.0/lib64:${LD_LIBRARY_PATH}
#    export PATH=/usr/local/cuda-7.0/bin:${PATH}
#    export CUDA_HOME=/usr/local/cuda
#    export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}
#    export LD_LIBRARY_PATH=/usr/local/cuda-6.5/lib64:${LD_LIBRARY_PATH}
#    export PATH=/usr/local/cuda-6.5/bin:${PATH}
#    export CUDA_BIN_PATH=/usr/local/cuda-7.5/
#    export PATH=/playpen/poirson/torch/bin:$PATH
#    export LD_LIBRARY_PATH=/playpen/poirson/cudnn/cuda/lib64/:$LD_LIBRARY_PATH
#
#    . /playpen/poirson/torch/install/bin/torch-activate
#
#
#
#
#
#
#
#
#fi
#
#
#
#
#



#export PATH=/usr/local/cuda-8.0/bin${PATH:+:${PATH}}
#export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64\
#                        ${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

#/usr/local/cuda-8.0/bin


#bvisionserver3 ????
#export PATH=/usr/local/cuda-8.0-cudnn-5.1/bin:${PATH}
#export LD_LIBRARY_PATH=/usr/local/cuda-8.0-cudnn-5.1/lib64:${LD_LIBRARY_PATH}
#export CUDA_HOME=/usr/local/cuda-8.0-cudnn-5.1
export PATH=/usr/local/cuda-8.0-cudnn-6.0/bin:${PATH}
export LD_LIBRARY_PATH=/usr/local/cuda-8.0-cudnn-6.0/lib64:${LD_LIBRARY_PATH}
export CUDA_HOME=/usr/local/cuda-8.0-cudnn-6.0
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${CUDA_HOME}/extras/CUPTI/lib64
#export PYTHONPATH=${PYTHONPATH}:/playpen/ammirato/target_driven_detection/code/exploring_nerual_networks/exploring_pytorch/basic_examples/faster_rcnn_pytorch/
#export PYTHONPATH=${PYTHONPATH}:/playpen/ammirato/code/
#export PYTHONPATH=${PYTHONPATH}:/playpen/ammirato/target_driven_detection/code/exploring_nerual_networks/
#export PYTHONPATH=${PYTHONPATH}:/playpen1/ammirato/target_driven_detection/code/exploring_nerual_networks/exploring_pytorch/basic_examples/faster_rcnn_pytorch/
#export PYTHONPATH=${PYTHONPATH}:/playpen1/ammirato/code/
#export PYTHONPATH=${PYTHONPATH}:/playpen1/ammirato/target_driven_detection/code/exploring_nerual_networks/
export PYTHONPATH=${PYTHONPATH}:/net/bvisionserver3/playpen/ammirato/target_driven_detection/code/exploring_nerual_networks/exploring_pytorch/basic_examples/faster_rcnn_pytorch/
export PYTHONPATH=${PYTHONPATH}:/net/bvisionserver3/playpen/ammirato/code/
export PYTHONPATH=${PYTHONPATH}:/net/bvisionserver3/playpen/ammirato/target_driven_detection/code/exploring_nerual_networks/
#export PYTHONPATH=${PYTHONPATH}:/net/bvisionserver3/playpen/ammirato/target_driven_detection/code/instance_detection/
export PYTHONPATH=${PYTHONPATH}:/net/bvisionserver3/playpen/ammirato/target_driven_detection/code/
export PYTHONPATH=${PYTHONPATH}:/net/bvisionserver3/playpen/ammirato/code/active_vision_dataset_processing/
export PYTHONDONTWRITEBYTECODE=1



# added by Anaconda2 4.4.0 installer
#export PATH="/playpen/ammirato/pytorch_vitrualenv/anaconda2/bin:$PATH"






#bvisionserver2, target_driven detection virtualenv
#export PATH=/usr/local/cuda-8.0.old/bin:${PATH}
#export LD_LIBRARY_PATH=/usr/local/cuda-8.0.old/lib64:${LD_LIBRARY_PATH}
#export CUDA_HOME=/usr/local/cuda-8.0.old
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${CUDA_HOME}/extras/CUPTI/lib64


