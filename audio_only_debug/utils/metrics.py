"""
Author: Smeet Shah
Copyright (c) 2020 Smeet Shah
File part of 'deep_avsr' GitHub repository available at -
https://github.com/lordmartian/deep_avsr
"""

import torch
import numpy as np
import editdistance
from config import args


def compute_cer(predictionBatch, targetBatch, predictionLenBatch, targetLenBatch):

    """
    Function to compute the Character Error Rate using the Predicted character indices and the Target character
    indices over a batch.
    CER is computed by dividing the total number of character edits (computed using the editdistance package)
    with the total number of characters (total => over all the samples in a batch).
    The <EOS> token at the end is excluded before computing the CER.
    """

    targetBatch = targetBatch.cpu()
    targetLenBatch = targetLenBatch.cpu()

    preds = list(torch.split(predictionBatch, predictionLenBatch.tolist()))
    trgts = list(torch.split(targetBatch, targetLenBatch.tolist()))
    totalEdits = 0
    totalChars = 0

    for n in range(len(preds)):
        pred = preds[n].numpy()[:-1]
        trgt = trgts[n].numpy()[:-1]
        pred = [int(x) for x in pred]
        trgt = [int(x) for x in trgt]
        numEdits = editdistance.eval(pred, trgt)
        totalEdits = totalEdits + numEdits
        totalChars = totalChars + len(trgt)

    return totalEdits/totalChars



def compute_wer(predictionBatch, targetBatch, predictionLenBatch, targetLenBatch, spaceIx):

    """
    Function to compute the Word Error Rate using the Predicted character indices and the Target character
    indices over a batch. The words are obtained by splitting the output at spaces.
    WER is computed by dividing the total number of word edits (computed using the editdistance package)
    with the total number of words (total => over all the samples in a batch).
    The <EOS> token at the end is excluded before computing the WER. Words with only a space are removed as well.
    """

    targetBatch = targetBatch.cpu()
    targetLenBatch = targetLenBatch.cpu()
    print("Walking through and example...")

    preds = list(torch.split(predictionBatch, predictionLenBatch.tolist()))
    trgts = list(torch.split(targetBatch, targetLenBatch.tolist()))

    # print("Predictions " + str(preds))
    # print("Targets " + str(trgts))
    totalEdits = 0
    totalWords = 0
    total_words = 0
    total_errors = 0
    index_to_char = args["INDEX_TO_CHAR"]
    for i in range(len(predictionBatch)):
        # Convert character predictions to words
        prediction = predictionBatch[i][:predictionLenBatch[i]]
        target = targetBatch[i][:targetLenBatch[i]]

        prediction_words = ''.join([chr(c) if c != spaceIx else ' ' for c in prediction]).split()
        target_words = ''.join([chr(c) if c != spaceIx else ' ' for c in target]).split()

        print(" prediction_words " + str(prediction_words))
        print(" target_words " + str(target_words))

        # Compute edit distance between prediction and target words
        errors = editdistance(prediction_words, target_words)

        total_errors += errors
        total_words += len(target_words)

    # Compute word error rate
    wer = total_errors / total_words if total_words > 0 else 0



    # for n in range(len(preds)):
    #     print("Walking through and example...")
    #     pred = preds[n].numpy()[:-1]
    #     trgt = trgts[n].numpy()[:-1]
    #
    #     #TURN TO INTS
    #     pred = [int(x) for x in pred]
    #     trgt = [int(x) for x in trgt]
    #
    #
    #     print("Prediction " + str(pred))
    #     print("Target " + str(trgt))
    #
    #     print("Trying something out")
    #     pred_indx = [index_to_char[x] for x in pred]
    #     print("pred words " + str(pred_indx))
    #
    #     targ_indx = [index_to_char[x] for x in trgt]
    #     print("targ words " + str(targ_indx))
    #
    #     print("editditance " + str(editdistance.eval(pred_indx, targ_indx)))
    #
    #     predWords = np.split(pred, np.where(pred == spaceIx)[0])
    #     predWords = [predWords[0].tostring()] + [predWords[i][1:].tostring() for i in range(1, len(predWords)) if len(predWords[i][1:]) != 0]
    #
    #
    #     trgtWords = np.split(trgt, np.where(trgt == spaceIx)[0])
    #     trgtWords = [trgtWords[0].tostring()] + [trgtWords[i][1:].tostring() for i in range(1, len(trgtWords))]
    #
    #     # print("Prediction words " + str(predWords))
    #     # print("Target words " + str(trgtWords))
    #
    #     numEdits = editdistance.eval(predWords, trgtWords)
    #     print("PREDICTED ONE " + str(numEdits))
    #     exit(1)
    #
    #     totalEdits = totalEdits + numEdits
    #     totalWords = totalWords + len(trgtWords)

    return totalEdits/totalWords
