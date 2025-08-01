# Timm requirements

## Transform usage

```python
data_cfg = timm.data.resolve_data_config(model.pretrained_cfg)
transform = timm.data.create_transform(**data_cfg)
transform
```

## Model loading

```python
import timm

m = timm.create_model('mobilenetv3_large_100', pretrained=True)
```