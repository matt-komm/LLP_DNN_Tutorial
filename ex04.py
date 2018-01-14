import tensorflow as tf
import os
import time
from root_reader import root_reader

fileList = [
    "data/qcd_1.root",
    "data/qcd_2.root",
    "data/qcd_3.root",
    "data/qcd_4.root",
    "data/ttbar_1.root",
    "data/ttbar_2.root",
    "data/ttbar_3.root",
    "data/ttbar_4.root"
]

featureDict = {
    "truth": {
        "branches":[
            'isB/UInt_t',
            'isBB/UInt_t',
            'isGBB/UInt_t',
            'isLeptonicB/UInt_t',
            'isLeptonicB_C/UInt_t',
            'isC/UInt_t',
            'isCC/UInt_t',
            'isGCC/UInt_t',
            'isUD/UInt_t',
            'isS/UInt_t',
            'isG/UInt_t',
            'isUndefined/UInt_t',
        ],
    },
    "globals": {
        "branches": [
            'jet_pt',
            'jet_eta',
            'nCpfcand',
            'nNpfcand',
            'nsv',
            'npv',
            'TagVarCSV_trackSumJetEtRatio', 
            'TagVarCSV_trackSumJetDeltaR', 
            'TagVarCSV_vertexCategory', 
            'TagVarCSV_trackSip2dValAboveCharm', 
            'TagVarCSV_trackSip2dSigAboveCharm', 
            'TagVarCSV_trackSip3dValAboveCharm', 
            'TagVarCSV_trackSip3dSigAboveCharm', 
            'TagVarCSV_jetNSelectedTracks', 
            'TagVarCSV_jetNTracksEtaRel'
        ],
    },
    "sv" : {
        "branches":[
            'sv_pt',
            'sv_deltaR',
            'sv_mass',
            'sv_ntracks',
        ],
        "max":4
    },
}


for epoch in range(1):
    print "epoch",epoch+1
    fileListQueue = tf.train.string_input_producer(
        fileList, 
        num_epochs=1, 
        shuffle=True
    )

    rootreader_op = [
        root_reader(
            fileListQueue, 
            featureDict,
            "deepntuplizer/tree",
            batch=1
        ).batch() for _ in range(1)
    ]
    print rootreader_op
    
    batchSize = 1
    minAfterDequeue = batchSize*2
    capacity = minAfterDequeue + 3 * batchSize
    
    trainingBatch = tf.train.shuffle_batch_join(
        rootreader_op, 
        batch_size=batchSize, 
        capacity=capacity,
        min_after_dequeue=minAfterDequeue,
        enqueue_many=True
    )
    
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess = tf.Session()
    sess.run(init_op)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    steps = 1
    try:
        while(True):
            t = time.time()
            result = sess.run(trainingBatch)
            print result
            t = time.time()-t
            print "step %3i (%8.3fs)"%(steps,t)
            steps+=1
            if (steps>10):
                break
    except tf.errors.OutOfRangeError:
        print "done"

    coord.request_stop()
    coord.join(threads)

