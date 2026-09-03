# Deep Learning HW3: Convolutional Neural Networks and Object Detection

This experiment studies convolutional neural networks for image and one-dimensional signal classification, focusing on kernel size, stride, padding, learned features, and gradient behavior.
It also analyzes a pretrained Faster R-CNN object detector to understand feature extraction, region proposals, receptive fields, anchors, and multi-scale object detection.
Additional experiments compare CNNs with MLPs and evaluate the effects of noise, transfer learning, filter depth, and dropout on accuracy, efficiency, and feature robustness.

Datasets: MNIST handwritten digits, synthetic noisy sine/cosine signals, and selected object-detection images; Models: 2D CNNs, 1D CNNs, MLPs, ResNet-18/ResNet-50 backbones, and Faster R-CNN with FPN.

## Experiments

### Experiment 1: 2D CNN Kernel Size on MNIST
**Description**
The first convolutional layer was evaluated with kernel sizes 1×1, 3×3, 5×5, and 7×7 while keeping stride 1, same padding, and 16 filters fixed.
Test accuracy was compared after 20 training epochs to determine which receptive-field size provides the best balance for 28×28 MNIST images.

**Achievements and Results**
The 5×5 kernel achieved the highest test accuracy at 99.1%, followed by 7×7 at 99.01%, 3×3 at 98.8%, and 1×1 at 98.47%.
The 5×5 kernel provided a useful receptive field for digit shapes while using fewer parameters than 7×7 and capturing more spatial context than 1×1.

### Experiment 2: Parameter Count versus 2D Kernel Size
**Description**
The number of trainable parameters was calculated for the CNN configurations used in the kernel-size experiment.
The comparison was used to determine how increasing the first-layer kernel changes model size while the remaining architecture stays fixed.

**Achievements and Results**
The models had similar parameter counts of roughly 20,000, but parameter count increased with kernel size.
The 7×7 configuration had 21,130 trainable parameters, compared with 20,362 for the 1×1 configuration.

### Experiment 3: Learned First-Layer Filters on MNIST
**Description**
The learned filters of the best-performing 5×5 configuration were visualized as matrices to inspect whether individual kernels had developed interpretable low-level detectors.
The filter patterns were examined for directional structures, edges, gradients, and other localized responses.

**Achievements and Results**
Clear classical edge-detector patterns were not dominant, but several filters showed smooth gradients and bright/dark structures.
Some filters responded to diagonal or horizontal lines, while several others appeared specialized but had not fully converged after 20 epochs.

### Experiment 4: First-Layer Feature Maps on a Test Digit
**Description**
A test image containing the digit 7 was passed through the first convolutional layer, and all 16 feature maps were visualized.
Each activation map was interpreted according to the local pattern emphasized by its corresponding filter.

**Achievements and Results**
Several filters strongly highlighted the diagonal stroke of the 7, while filters 14, 11, and 7 responded more strongly to its horizontal stroke.
Other filters emphasized bright points or parts of the digit without a clearly interpretable semantic pattern for this specific input.

### Experiment 5: Stride Effect on Output Size and Model Complexity
**Description**
A fixed CNN with 3×3 convolutions, 32 and 64 filters, and 2×2 pooling was tested with first-layer stride values of 1 and 2.
Padding was selected so that the post-convolution output was either equal to the input size or approximately half of it.

**Achievements and Results**
Stride 1 produced a 28×28 feature map and 50,186 trainable parameters, while stride 2 produced a 14×14 map and 24,586 parameters.
Training time was 283 seconds versus 274 seconds for 20 epochs, showing a large reduction in parameters but only about a 3% runtime difference.

### Experiment 6: Stride Effect on MNIST Accuracy
**Description**
The same stride configurations were evaluated using final test accuracy to determine whether the spatial downsampling introduced by stride 2 harms classification performance.
The comparison used the same architecture and training schedule so that stride was the main changing factor.

**Achievements and Results**
Stride 1 achieved 99.2% test accuracy, while stride 2 achieved 98.64%, indicating a small but measurable performance loss.
The higher accuracy of stride 1 was attributed to retaining more fine-grained edge information, whereas stride 2 reduced parameters and remained competitive.

### Experiment 7: Valid versus Same Padding on MNIST
**Description**
The CNN was evaluated with valid padding and same padding at stride 1 to study the effect of spatial padding on tensor dimensions and classification.
The theoretical reduction in spatial size under repeated valid convolutions was also used to analyze depth limitations in deeper networks.

**Achievements and Results**
With a 3×3 kernel and valid padding, a 28×28 image reaches zero spatial size after 14 consecutive convolutional layers without pooling, losing two pixels per layer.
Same padding preserves spatial dimensions and allows substantially deeper convolutional stacks before spatial collapse, while also preserving information near image boundaries.

### Experiment 8: Padding Effect on Final Accuracy
**Description**
Final test accuracy was compared between valid and same padding configurations while keeping the rest of the CNN and stride fixed.
The goal was to determine whether preserving spatial dimensions produces a meaningful improvement on the small MNIST images.

**Achievements and Results**
The valid-padding model achieved 98.64% accuracy and the same-padding model achieved 98.58%, a negligible difference of 0.06 percentage points.
For this simple dataset and relatively shallow network, the reduced spatial representation was sufficient, making padding choice only weakly influential on accuracy.

### Experiment 9: Weight Statistics during CNN Training
**Description**
The selected CNN was trained for 20 epochs while the mean and standard deviation of the first convolutional layer weights were recorded after every epoch.
The evolution of these statistics was used to assess whether the weights remained stable and whether gradient-related instability was visible.

**Achievements and Results**
Weights started near zero with a small standard deviation and gradually moved toward larger magnitudes as training progressed.
The observed gradient behavior did not indicate severe vanishing or exploding gradients, consistent with the use of ReLU, Adam, and standard random initialization.

### Experiment 10: Weight Distribution before and after Training
**Description**
Histograms of the first convolutional layer weights were compared at the beginning and end of training to visualize how optimization changed the parameter distribution.
The comparison focused on changes in concentration, spread, and distance from zero as the model learned useful features.

**Achievements and Results**
The initial distribution was concentrated near zero, while the final distribution became broader and extended to larger positive and negative values, approximately within −0.5 to 0.5.
This widening suggests that the filters moved away from their initialization and adapted to meaningful image structures, although complete convergence had not yet been reached.

### Experiment 11: Vanishing Gradients with Sigmoid Activations
**Description**
A deep 10-layer CNN without residual connections or batch normalization was trained on MNIST using Sigmoid activations, and the gradient norms of the first and last convolutional layers were tracked during training.
The experiment tested whether gradients in early layers become substantially smaller than those near the output.

**Achievements and Results**
The first-layer gradient norm became many orders of magnitude smaller than the last-layer norm and approached zero over time, demonstrating clear vanishing-gradient behavior.
The network therefore learned much more slowly in early layers, with gradient attenuation caused by repeated multiplication through saturated Sigmoid derivatives.

### Experiment 12: ReLU versus Sigmoid Gradient Stability
**Description**
The same deep CNN experiment was repeated with ReLU activations instead of Sigmoid to isolate the effect of the activation function on gradient propagation.
Gradient norms from the first and last convolutional layers were compared throughout the 20-epoch training process.

**Achievements and Results**
With ReLU, the gap between early- and late-layer gradient norms was much smaller and the gradients stabilized more consistently than with Sigmoid.
The first-layer gradient remained around 0.1 while the last layer stayed between approximately 0 and 1, showing substantially healthier propagation.

### Experiment 13: Exploding Gradients with Large Initialization
**Description**
The deep Sigmoid CNN was initialized with weights scaled by a factor of 10 to deliberately create an unstable optimization regime.
Gradient norms were monitored to determine whether the enlarged initialization causes rapid growth and numerical instability.

**Achievements and Results**
Gradient norms became extremely large, reaching values around 10,000 during the early stages of training and causing severe numerical instability.
The resulting explosion can drive the loss toward NaN or divergence, while the first-layer gradients remained much smaller than the last-layer gradients because of Sigmoid attenuation.

### Experiment 14: 1D CNN Kernel Size on Noisy Signals
**Description**
A 1D convolutional network for length-100 sine/cosine signals was evaluated with first-layer kernel sizes 3, 5, 7, and 11 using stride 1, same padding, and 8 filters.
The initial low-noise setting produced uniformly high accuracy, so additional frequency, phase, and stronger noise variation was introduced to make the comparison more informative.

**Achievements and Results**
Under the harder noisy setting, kernel size 3 achieved 98.5% test accuracy, followed by 5 at 98%, 7 at 97.5%, and 11 at 97%.
Small kernels worked better because the discriminative changes between sine and cosine occur over relatively short local regions of the 100-sample signals.

### Experiment 15: Very Large 1D Kernel
**Description**
A much larger kernel, including size 51, was tested to examine whether covering roughly half a signal cycle destroys local information needed to distinguish sine from cosine.
Both theoretical expectations and direct experiments with large kernels such as 21, 31, 41, 51, and 99 were considered.

**Achievements and Results**
The theoretical analysis predicted strong smoothing and reduced separability, but the size-51 experiment still achieved about 96.5% accuracy, with other very large kernels also remaining above 90%.
The unexpected robustness was attributed to fixed phase patterns, learnable biases, multiple convolutional layers, and the ability of later layers to compensate for the coarse first-layer response.

### Experiment 16: 1D CNN Stride and Padding
**Description**
Stride and padding were varied for the 1D CNN to determine how temporal downsampling changes output length, parameter count, and classification accuracy.
Configurations included same and valid padding with stride 1 and stride 2, followed by a direct comparison with the 2D CNN behavior.

**Achievements and Results**
The best result was same padding with stride 1 at 98.5% accuracy and output length 100, while valid padding with stride 1 reached 98% with output length 98.
Stride 2 reduced the output length to 50 and the parameter count to 626, but accuracy dropped to 95.5% with valid padding and about 92% with the reported same-padding calculation.

### Experiment 17: 1D CNN Feature Maps
**Description**
Feature maps from the first convolutional layer were visualized for test sine and cosine signals to identify kernels behaving like derivative filters, smoothers, or other temporal pattern detectors.
The activation shapes were compared with the input waveform to interpret what temporal structures individual filters emphasized.

**Achievements and Results**
Feature maps 2, 5, and 6 behaved similarly to derivative operators, responding strongly to steep local slopes in the input signal.
Feature maps 1, 3, 4, and 7 behaved more like smoothers, while map 8 appeared to reflect the approximate mean level of the input signal.

### Experiment 18: Faster R-CNN Object Detection on Selected Images
**Description**
A pretrained Faster R-CNN model with a ResNet-50 backbone and FPN was applied to three selected images containing a car, two cats, and multiple birds.
Detected objects, bounding boxes, and confidence scores were inspected to evaluate practical object-detection performance and false detections.

**Achievements and Results**
The car was detected with 100% confidence, the two cats with 100% and 99%, and the birds with confidence scores of 100%, 100%, 99%, 93%, 89%, and 70%.
One bird was detected twice at 89% and 70%; the duplicate was attributed to the bird's unusual pose with extended wings and broad silhouette rather than a fundamentally incorrect object category.

### Experiment 19: ResNet-50 Initial Kernel and Receptive Field
**Description**
The first 7×7 convolutional kernel of the ResNet-50 backbone was analyzed against the later 3×3 convolutions to understand its role in early receptive-field growth.
The comparison considered contextual coverage, computational depth, and the amount of spatial information available to the first learned representation.

**Achievements and Results**
The 7×7 kernel provides a larger initial receptive field, capturing broader edges, textures, and coarse context earlier than a single 3×3 layer.
Two stacked 3×3 convolutions are needed to obtain a comparable receptive field, so a single 7×7 layer can provide wider context with a shallower initial stage.

### Experiment 20: High Effective Stride and Small Object Detection
**Description**
The effect of an effective backbone stride of 32 was analyzed in terms of spatial resolution and the representation of small objects.
The role of Feature Pyramid Networks was considered as a mechanism for recovering useful high-resolution features for small-object detection.

**Achievements and Results**
At stride 32, one feature-map pixel corresponds to roughly a 32×32 region of the input, so very small objects may shrink to less than one meaningful feature-map pixel.
FPN addresses this limitation by combining feature maps at strides 4, 8, 16, and 32 so that small objects can be detected on higher-resolution levels.

### Experiment 21: Intermediate ResNet Feature Maps
**Description**
An input image was passed through an intermediate ResNet-50 layer and multiple feature-map channels were visualized to determine whether specific filters respond to recognizable object parts.
The analysis focused on edges, textures, vehicle components, and other localized structures in the selected images.

**Achievements and Results**
For the car image, channel 12 highlighted wheels and headlights, channel 15 responded to a horizontal or diagonal structure below the vehicle, and channel 9 highlighted the curved roof surface.
Other channels responded to vehicle texture or showed no clearly identifiable activation on the selected input.

### Experiment 22: Multi-Scale Detection with Stride, Pooling, and Anchor Boxes
**Description**
The mechanism by which Faster R-CNN detects objects at different positions and scales was analyzed using convolutional stride, pooling, RPN anchor boxes, and multi-level feature representations.
The explanation connected local convolutional receptive fields to progressively larger contextual fields and region proposals.

**Achievements and Results**
Stride and pooling progressively reduce spatial resolution while increasing receptive fields, allowing deeper layers to represent broader object context while early layers preserve local detail.
RPN anchor boxes of different sizes and aspect ratios provide scale-aware templates, while FPN combines multiple resolutions so both small and large objects can be localized.

### Experiment 23: CNN versus MLP on MNIST
**Description**
A two-hidden-layer MLP with 512 and 256 neurons was compared with a simple two-convolution CNN on MNIST using test accuracy, parameter count, and training behavior.
The comparison was designed to evaluate the benefits of local connectivity, weight sharing, and hierarchical feature extraction for image data.

**Achievements and Results**
The MLP contained about 535,000 parameters compared with about 50,000 for the CNN, while final test accuracy was 98.07% for the MLP and 99.11% for the CNN.
The CNN converged in about 2 epochs versus about 5 for the MLP and achieved better accuracy despite its much smaller parameter count.

### Experiment 24: CNN Robustness to Gaussian Noise
**Description**
Gaussian noise with standard deviations 0.2 and 0.5 was added to MNIST and the CNN was retrained from scratch on the noisy datasets.
Final test accuracy and first-layer feature maps were compared with the clean-data baseline to determine whether convolutional filtering suppresses part of the noise.

**Achievements and Results**
The CNN achieved 99.07% accuracy with noise standard deviation 0.2 and 99.05% with standard deviation 0.5, compared with 99.11% on clean MNIST.
The small accuracy decrease suggests that convolutional filters can suppress some high-frequency noise, although stronger noise still appears in feature maps and slightly reduces performance.

### Experiment 25: Transfer Learning from ImageNet to MNIST
**Description**
A pretrained ResNet-18 model trained on ImageNet was adapted by replacing its final classifier with a 10-class MNIST output and fine-tuning the network.
The transfer-learning result was compared with the CNN trained from scratch to determine whether generic visual features remain useful across substantially different domains.

**Achievements and Results**
The fine-tuned ResNet-18 reached a best test accuracy of 97.0% and 96.85% at the final training epoch despite changing only the final layer.
Transfer learning was still useful because early ImageNet features such as edges, simple textures, and basic patterns remained transferable to handwritten digits.

### Experiment 26: ResNet First-Layer Weight Stability during Fine-Tuning
**Description**
The first convolutional-layer weights of the pretrained ResNet-18 were compared before and after fine-tuning to quantify how much low-level representations changed.
The experiment focused on whether adaptation to MNIST requires substantial modification of the earliest visual filters.

**Achievements and Results**
When only deeper layers are fine-tuned, the first-layer weights remain unchanged, indicating that generic low-level ImageNet features can be reused directly.
Even when more of the network is fine-tuned, the earliest filters are expected to change only slightly because they encode broadly useful edges, gradients, and textures.

### Experiment 27: Deep Sigmoid CNN Training Failure
**Description**
A 15-layer CNN without batch normalization or residual connections was trained on MNIST with Sigmoid activations to test whether a very deep convolutional stack can learn effectively.
Training loss and test accuracy were monitored over 20 epochs, with particular attention to early-layer gradient attenuation.

**Achievements and Results**
The network failed to learn meaningful representations, reaching only 11.35% test accuracy after 20 epochs and showing no substantial reduction in training loss.
The failure was linked to vanishing gradients in early layers caused by repeated multiplication through the small derivatives of saturated Sigmoid activations.

### Experiment 28: Deep Sigmoid CNN versus ReLU plus Batch Normalization
**Description**
Gradient norms at the first and last layers were compared between the deep Sigmoid network and a corresponding network using ReLU with batch normalization.
The comparison evaluated gradient magnitude, relative scaling between early and late layers, and the resulting classification performance.

**Achievements and Results**
For Sigmoid, the last-layer gradient was about 10,000 times larger than the first at the beginning, and the first-layer gradient became effectively zero by the end of training.
With ReLU and batch normalization, the two gradient norms were of similar scale and remained between approximately 0.001 and 1, while test accuracy reached 99.44%.

### Experiment 29: Effect of Filter Count on CNN Capacity
**Description**
A fixed two-layer MNIST CNN was evaluated with filter pairs (4,8), (8,16), (16,32), (32,64), and (64,128) while keeping Adam, learning rate 0.001, batch size 64, and 20 epochs fixed.
The experiment measured trainable parameters, final test accuracy, and total training time to study the effect of channel depth on model capacity.

**Achievements and Results**
Increasing the number of filters generally increased parameter count, training time, and test accuracy, with the highest accuracy obtained by the (64,128) configuration.
However, the reported curves suggested diminishing returns, indicating that much larger models add parameters without a proportional gain on the relatively simple MNIST task.

### Experiment 30: Best Filter Configuration by Accuracy–Size Trade-off
**Description**
The tested filter configurations were compared using test accuracy together with model size to identify a practical compromise between predictive performance and parameter count.
The selected configuration was chosen to remain close to the best accuracy while avoiding the largest parameter increase.

**Achievements and Results**
The (32,64) configuration was selected as the best balance, with 20,490 parameters and 98.87% test accuracy.
Its accuracy was only 0.05 percentage points below the reported best result while requiring substantially fewer parameters than the largest configuration.

### Experiment 31: Feature Maps with Low versus High Filter Counts
**Description**
First-layer feature maps were compared for the (4,8) and (64,128) CNN configurations using the same MNIST test digit.
The comparison examined whether increasing channel depth produces more specialized filters and more detailed representations.

**Achievements and Results**
The low-filter model captured the main structure of the digit 7 but several filters appeared to combine multiple simple features rather than specialize strongly.
The high-filter model learned more complex and specific detectors for horizontal, diagonal, and thick-line structures, although some appeared overly specialized and potentially closer to memorizing patterns.

### Experiment 32: Dropout for Noise Robustness in CNNs
**Description**
A fixed CNN was trained with dropout rates 0, 0.2, 0.5, and 0.7 on MNIST with Gaussian noise of standard deviation 0.4, while evaluation was performed on both clean and similarly noisy test sets.
Final clean accuracy, noisy accuracy, and the clean-to-noisy accuracy gap were compared for each dropout rate.

**Achievements and Results**
Dropout 0.5 produced the highest clean test accuracy at 99.38% and 99.11% on noisy data, while dropout 0.2 gave the highest noisy accuracy at 99.12%.
The largest clean/noisy accuracy gap was 0.27% for dropout 0.5, while dropout 0 had the smallest gap, showing that accuracy alone does not fully capture robustness differences.

### Experiment 33: Dropout Rate and Convergence Speed under Noise
**Description**
Test accuracy curves for dropout rates 0, 0.2, 0.5, and 0.7 were compared throughout training on the noisy MNIST setting.
The goal was to identify which regularization level reaches a strong accuracy region most quickly while maintaining stable performance.

**Achievements and Results**
The four configurations followed broadly similar convergence trajectories, making the exact fastest configuration difficult to distinguish visually.
Dropout 0 started with higher accuracy and was judged to converge fastest overall, while dropout 0.2 followed as the fastest regularized configuration.

### Experiment 34: Dropout and Overfitting under Noise
**Description**
The difference between training and noisy-test accuracy was examined for each dropout rate to determine whether increasing dropout reduces overfitting.
The best dropout rate was selected using the highest noisy-test accuracy while also considering the training/test gap.

**Achievements and Results**
Dropout 0.2 produced the highest noisy-test accuracy at 99.12%, and all configurations had training-to-test differences below 1%.
The largest reported gap was 0.7% for dropout 0, supporting the conclusion that an appropriate dropout rate reduces overfitting.

### Experiment 35: Best Dropout Model on Clean MNIST
**Description**
The best dropout configuration from the noisy-data experiment was retrained from scratch on clean MNIST to determine whether its regularization remains effective without input corruption.
Clean-data performance was compared with the corresponding noisy-data behavior.

**Achievements and Results**
The selected dropout model also performed strongly on clean data, with both clean and noisy settings maintaining high and closely matched accuracy.
This indicates that the learned representation generalized well and that the regularization did not prevent the model from fitting the clean MNIST task.

### Experiment 36: Dropout Effect on Feature Noise
**Description**
A noisy test image containing the digit 7 was passed through two CNNs, one with dropout 0 and another with dropout 0.5, and their first-layer feature maps were visualized.
The activation patterns were compared to determine whether dropout changes the extent to which filters respond to noise versus meaningful digit structure.

**Achievements and Results**
The dropout-0 model contained filters that clearly activated on noisy regions, including one filter that learned substantial parts of the noise pattern.
The dropout-0.5 model produced several more meaningful line and stroke detectors, suggesting that dropout encouraged filters to become more resistant to noise and focus on robust digit features.
