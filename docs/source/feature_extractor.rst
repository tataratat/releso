Feature Extractor
=================

This package has (Version>=1.1) the ability to use custom feature
extractors for the agent networks. When using a custom feature extractor the
observations are first fed into the feature extractor and the result is then
fed into the agent networks. Please see the `stable-baselines3 
<https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html#custom-feature-extractor>`_
section on feature extractors for more information on how this works. 

This library is now also able to use more complex observation spaces, like 
image based observation spaces and observation spaces which are made up of 
scalar observations (normal observations) and image observations.

Both functionalities come from the feature extractors. Please see the class 
definitions of :py:class:`SbSOvRL.feature_extractor.SbSOvRL_FeatureExtractor` 
and :py:class:`SbSOvRL.feature_extractor.SbSOvRL_CombinedExtractor` for the 
possible parameters.

The image based observations are explained in more detail on the page 
:doc:`Image based Observations<image_based_observations>`.