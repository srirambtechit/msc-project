# msc-project

### Interpreting detectron2 prediction result
![Predicted image](./output/predicted_image.png)
```
{
  'instances': Instances(
    num_instances=2, 
    image_height=416, 
    image_width=416, 
    fields=[
      pred_boxes: Boxes(
        tensor([
          [ 95.2393,  49.6693, 233.6631, 362.2609],
          [301.9677,  33.7873, 408.4108, 380.9571]
        ], device='cuda:0')),
        scores: tensor([0.9685, 0.9662], device='cuda:0'), 
        pred_classes: tensor([3, 1], device='cuda:0')
    ]
  )
}
```
