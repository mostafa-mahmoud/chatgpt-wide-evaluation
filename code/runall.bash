#!/bin/bash

problems="sarcasm subjectivity engagement toxicity sadness_emotion_intensity joy_emotion_intensity fear_emotion_intensity anger_emotion_intensity well_being_reddit_body well_being_reddit well_being_reddit_titles well_being_twitter well_being_twitter_full aspect_res14_target aspect_lap14_target aspect_res15_target aspect_res14_polarity aspect_lap14_polarity aspect_res15_polarity aspect_res14_opinion aspect_lap14_opinion aspect_res15_opinion"
list0="sarcasm subjectivity engagement sadness_emotion_intensity joy_emotion_intensity fear_emotion_intensity anger_emotion_intensity"
list1="well_being_reddit_body well_being_reddit well_being_reddit_titles well_being_twitter well_being_twitter_full toxicity"
list2="aspect_res14_target aspect_lap14_target aspect_res15_target aspect_res14_polarity aspect_lap14_polarity aspect_res15_polarity aspect_res14_opinion aspect_lap14_opinion aspect_res15_opinion"

#list2="aspect_res14_target aspect_lap14_target aspect_res15_target aspect_res14_polarity aspect_lap14_polarity"
#list1="aspect_res15_polarity aspect_res14_opinion aspect_lap14_opinion aspect_res15_opinion"
#problems_ex="sarcasm subjectivity engagement sadness_emotion_intensity joy_emotion_intensity fear_emotion_intensity anger_emotion_intensity well_being_reddit_body well_being_reddit well_being_reddit_titles well_being_twitter well_being_twitter_full aspect_res14_target aspect_lap14_target aspect_res15_target aspect_res14_polarity aspect_lap14_polarity aspect_res15_polarity aspect_res14_opinion aspect_lap14_opinion aspect_res15_opinion"

for action in hptune predict ;
# for action in predict ;
do
    for features in BoW ;
    do
        if [ "$CUDA_VISIBLE_DEVICES" == "0" ]; then
            torun=$list0
        elif [ "$CUDA_VISIBLE_DEVICES" == "1" ]; then
            torun=$list1
        elif [ "$CUDA_VISIBLE_DEVICES" == "2" ]; then
            torun=$list2
        else
            echo "Invalid argument. Please provide 1, 2, or 3."
        fi
        # torun=$problems_ex
        for problem in $torun ;
        do
            echo "./run_experiment.py $action $features $problem"
            ./run_experiment.py $action $features $problem
            echo "========================================"
            echo "========================================"
            echo "========================================"
        done
    done
done

