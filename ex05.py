import tensorflow as tf
import keras
from keras import backend as K
import os
import time
from root_reader import root_reader

from deep_jet_model import model_deepFlavourReference

import imp
try:
    imp.find_module('setGPU')
    import setGPU
except ImportError:
    pass


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

     "sv" : {
        "branches":[
            'sv_pt',
            'sv_deltaR',
            'sv_mass',
            'sv_ntracks',
            'sv_chi2',
            'sv_normchi2',
            'sv_dxy',
            'sv_dxysig',
            'sv_d3d',
            'sv_d3dsig',
            'sv_costhetasvpv',
            'sv_enratio',
            
        ],
        "max":4
    },

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
    
    "global": {
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


    "Cpfcan": {
        "branches": [
            'Cpfcan_BtagPf_trackEtaRel',
            'Cpfcan_BtagPf_trackPtRel',
            'Cpfcan_BtagPf_trackPPar',
            'Cpfcan_BtagPf_trackDeltaR',
            'Cpfcan_BtagPf_trackPParRatio',
            'Cpfcan_BtagPf_trackSip2dVal',
            'Cpfcan_BtagPf_trackSip2dSig',
            'Cpfcan_BtagPf_trackSip3dVal',
            'Cpfcan_BtagPf_trackSip3dSig',
            'Cpfcan_BtagPf_trackJetDistVal',

            'Cpfcan_ptrel', 
            'Cpfcan_drminsv',
            'Cpfcan_VTX_ass',
            'Cpfcan_puppiw',
            'Cpfcan_chi2',
            'Cpfcan_quality'
        ],
        "max":25
    },
    "Npfcan": {
        "branches": [
            'Npfcan_ptrel',
            'Npfcan_deltaR',
            'Npfcan_isGamma',
            'Npfcan_HadFrac',
            'Npfcan_drminsv',
            'Npfcan_puppiw'
        ],
        "max":25
    }
}

for epoch in range(40):
    epoch_duration = time.time()
    print "epoch",epoch+1
    fileListQueue = tf.train.string_input_producer(fileList, num_epochs=1, shuffle=True)

    rootreader_op = [
        root_reader(fileListQueue, featureDict,"deepntuplizer/tree",batch=100).batch() for _ in range(6)
    ]
    
    batchSize = 1000
    minAfterDequeue = batchSize*2
    capacity = minAfterDequeue + 3*batchSize
    
    trainingBatch = tf.train.shuffle_batch_join(
        rootreader_op, 
        batch_size=batchSize, 
        capacity=capacity,
        min_after_dequeue=minAfterDequeue,
        enqueue_many=True #requires to read examples in batches!
    )

    globalvars = keras.layers.Input(tensor=trainingBatch['global'])
    cpf = keras.layers.Input(tensor=trainingBatch['Cpfcan'])
    npf = keras.layers.Input(tensor=trainingBatch['Npfcan'])
    vtx = keras.layers.Input(tensor=trainingBatch['sv'])
    truth = trainingBatch["truth"]

    nclasses = truth.shape.as_list()[1]
    inputs = [globalvars,cpf,npf,vtx]
    prediction = model_deepFlavourReference(inputs,nclasses,1,dropoutRate=0.1,momentum=0.6)
    loss = tf.reduce_mean(keras.losses.categorical_crossentropy(truth, prediction))
    accuracy,accuracy_op = tf.metrics.accuracy(tf.argmax(truth,1),tf.argmax(prediction,1))
    model = keras.Model(inputs=inputs, outputs=prediction)
    
    train_op = tf.train.AdamOptimizer(
        learning_rate=0.0001,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-08,
        use_locking=True,
        name='Adam'
    ).minimize(
        loss
    )
    
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    sess = K.get_session()

    sess.run(init_op)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    total_loss = 0
    
    if os.path.exists("model_epoch"+str(epoch-1)+".hdf5"):
        print "loading weights ... model_epoch"+str(epoch-1)+".hdf5"
        model.load_weights("model_epoch"+str(epoch-1)+".hdf5") #use after init_op which initializes random weights!!!
    elif epoch>0:
        print "no weights from previous epoch found"
        sys.exit(1)
        
    try:
        step = 0
        while not coord.should_stop():
            step += 1
            start_time = time.time()
            #Note: model.updates needed to update the mean/variance in batchnorm layers
            _, loss_value, accuracy_value,_ = sess.run([train_op, loss,accuracy_op, model.updates], feed_dict={K.learning_phase(): 1}) #pass 1 for training, 0 for testing
            total_loss+=loss_value
            duration = time.time() - start_time
            print 'Step %d: loss = %.2f, accuracy = %.1f%% (%.3f sec)' % (step, loss_value,accuracy_value*100.,duration)
    except tf.errors.OutOfRangeError:
        print('Done training for %d steps.' % (step))
    model.save_weights("model_epoch"+str(epoch)+".hdf5")
    print "Epoch duration = (%.1f min)"%((time.time()-epoch_duration)/60.)
    print "Average loss = ",(total_loss/step)
    f = open("model_epoch.stat","a")
    f.write(str(epoch)+";"+str(total_loss/step)+";"+str(accuracy_value*100.)+"\n")
    f.close()
    coord.request_stop()
    coord.join(threads)
    K.clear_session()
    
    
    
    
