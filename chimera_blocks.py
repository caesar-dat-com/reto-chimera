"""
chimera_blocks.py



Contiene:
    - Los 12 bloques (6 familias x 2 variantes: entrada / intermedia)
    - BloqueConvCustom, ChannelAdapter
    - ChimeraNet (arquitectura ensamblable)
    - Tablas de costos y calcular_presupuesto()
    - FocalLoss
    - Utilidades de entrenamiento, evaluacion F1 y firma anti-fraude
"""

import hashlib
import time
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score


# ---------------------------------------------------------------------------
# Bloques: familia por familia, variante "entrada" (mas simple) e "intermedia"
# (mas profunda / mas capacidad). Todos preservan el numero de canales excepto
# los DenseNet (crecen por concatenacion) y el Inception intermedio (duplica
# canales por tener bottlenecks menos agresivos) -- para esos, ChannelAdapter
# se encarga de devolver el ancho a base_channels.
# ---------------------------------------------------------------------------

class VGGBlockEntrada(nn.Module):
    """Dos convoluciones 3x3 + BN + ReLU, canales constantes."""
    def __init__(self, c):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c, c, 3, padding=1), nn.BatchNorm2d(c), nn.ReLU(inplace=True),
            nn.Conv2d(c, c, 3, padding=1), nn.BatchNorm2d(c), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class VGGBlockIntermedia(nn.Module):
    """Tres convoluciones 3x3 en vez de dos: mas profundidad, mismo patron."""
    def __init__(self, c):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c, c, 3, padding=1), nn.BatchNorm2d(c), nn.ReLU(inplace=True),
            nn.Conv2d(c, c, 3, padding=1), nn.BatchNorm2d(c), nn.ReLU(inplace=True),
            nn.Conv2d(c, c, 3, padding=1), nn.BatchNorm2d(c), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class ResNetBlockEntrada(nn.Module):
    """BasicBlock: dos convoluciones 3x3 con conexion de salto."""
    def __init__(self, c):
        super().__init__()
        self.conv1 = nn.Conv2d(c, c, 3, padding=1); self.bn1 = nn.BatchNorm2d(c)
        self.conv2 = nn.Conv2d(c, c, 3, padding=1); self.bn2 = nn.BatchNorm2d(c)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identidad = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identidad)


class ResNetBlockIntermedia(nn.Module):
    """BasicBlock extendido a tres convoluciones en la ruta residual.
    Mismos parametros que VGGBlockIntermedia -- la conexion de salto, otra vez,
    no cuesta nada en parametros."""
    def __init__(self, c):
        super().__init__()
        self.conv1 = nn.Conv2d(c, c, 3, padding=1); self.bn1 = nn.BatchNorm2d(c)
        self.conv2 = nn.Conv2d(c, c, 3, padding=1); self.bn2 = nn.BatchNorm2d(c)
        self.conv3 = nn.Conv2d(c, c, 3, padding=1); self.bn3 = nn.BatchNorm2d(c)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identidad = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        return self.relu(out + identidad)


class InceptionBlockEntrada(nn.Module):
    """Modulo Inception reducido (estilo GoogLeNet): 4 ramas con bottleneck 1x1
    agresivo (b=c/4). Es el bloque mas barato del catalogo: los bottlenecks
    reducen los canales antes de las convoluciones grandes."""
    def __init__(self, c):
        super().__init__()
        b = c // 4
        self.branch1 = nn.Sequential(nn.Conv2d(c, b, 1), nn.BatchNorm2d(b), nn.ReLU(inplace=True))
        self.branch2 = nn.Sequential(
            nn.Conv2d(c, b, 1), nn.BatchNorm2d(b), nn.ReLU(inplace=True),
            nn.Conv2d(b, b, 3, padding=1), nn.BatchNorm2d(b), nn.ReLU(inplace=True))
        self.branch3 = nn.Sequential(
            nn.Conv2d(c, b // 2, 1), nn.BatchNorm2d(b // 2), nn.ReLU(inplace=True),
            nn.Conv2d(b // 2, b, 5, padding=2), nn.BatchNorm2d(b), nn.ReLU(inplace=True))
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(c, b, 1), nn.BatchNorm2d(b), nn.ReLU(inplace=True))

    def forward(self, x):
        return torch.cat([self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)], dim=1)


class InceptionBlockIntermedia(nn.Module):
    """Version mas ancha: bottleneck mas suave (b=c/2 en vez de c/4). Al usar
    menos reduccion, la salida concatenada duplica los canales de entrada
    (2c en vez de c) -- ChimeraNet le agrega un ChannelAdapter automaticamente."""
    def __init__(self, c):
        super().__init__()
        b = c // 2
        self.branch1 = nn.Sequential(nn.Conv2d(c, b, 1), nn.BatchNorm2d(b), nn.ReLU(inplace=True))
        self.branch2 = nn.Sequential(
            nn.Conv2d(c, b, 1), nn.BatchNorm2d(b), nn.ReLU(inplace=True),
            nn.Conv2d(b, b, 3, padding=1), nn.BatchNorm2d(b), nn.ReLU(inplace=True))
        self.branch3 = nn.Sequential(
            nn.Conv2d(c, b // 2, 1), nn.BatchNorm2d(b // 2), nn.ReLU(inplace=True),
            nn.Conv2d(b // 2, b, 5, padding=2), nn.BatchNorm2d(b), nn.ReLU(inplace=True))
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(c, b, 1), nn.BatchNorm2d(b), nn.ReLU(inplace=True))

    def forward(self, x):
        return torch.cat([self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)], dim=1)


class DenseNetBlockEntrada(nn.Module):
    """Una capa densa: BN-ReLU-Conv1x1 (bottleneck) + BN-ReLU-Conv3x3, y la
    salida se concatena a la entrada (no la reemplaza): c canales entran,
    c+k salen, con k=growth rate."""
    def __init__(self, c, k=32):
        super().__init__()
        inter = 4 * k
        self.bn1 = nn.BatchNorm2d(c); self.conv1 = nn.Conv2d(c, inter, 1)
        self.bn2 = nn.BatchNorm2d(inter); self.conv2 = nn.Conv2d(inter, k, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        out = self.conv2(self.relu(self.bn2(out)))
        return torch.cat([x, out], dim=1)


class DenseNetBlockIntermedia(nn.Module):
    """Dos capas densas apiladas: la segunda recibe la concatenacion de la
    entrada original con la salida de la primera, tal como dentro de un
    bloque denso real con varias capas. c canales entran, c+2k salen."""
    def __init__(self, c, k=32):
        super().__init__()
        self.capa1 = DenseNetBlockEntrada(c, k)
        self.capa2 = DenseNetBlockEntrada(c + k, k)

    def forward(self, x):
        x = self.capa1(x)
        x = self.capa2(x)
        return x


class LayerNorm2d(nn.Module):
    """LayerNorm aplicado sobre el canal C en tensores (N,C,H,W) -- normaliza
    el vector de canales de cada posicion espacial de forma independiente,
    como en la implementacion oficial de ConvNeXt (channels_first)."""
    def __init__(self, c, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(c))
        self.bias = nn.Parameter(torch.zeros(c))
        self.eps = eps

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[None, :, None, None] * x + self.bias[None, :, None, None]


class ConvNeXtBlockEntrada(nn.Module):
    """Bloque ConvNeXt simplificado: conv depthwise 7x7 + LayerNorm + MLP
    invertido (expansion x2 aqui, x4 en la version intermedia) + GELU +
    layer-scale, con conexion de salto. Nota curricular: ConvNeXt se explica
    a fondo en la Semana 13 (requiere Vision Transformers y Swin); aqui se usa
    de forma puramente operacional -- se instancia y se combina, sin exigir
    la teoria de atencion que motiva su diseno."""
    def __init__(self, c, expansion=2):
        super().__init__()
        self.dwconv = nn.Conv2d(c, c, kernel_size=7, padding=3, groups=c)
        self.norm = LayerNorm2d(c)
        self.pwconv1 = nn.Conv2d(c, expansion * c, kernel_size=1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(expansion * c, c, kernel_size=1)
        self.gamma = nn.Parameter(1e-6 * torch.ones(c))

    def forward(self, x):
        identidad = x
        out = self.dwconv(x)
        out = self.norm(out)
        out = self.pwconv1(out)
        out = self.act(out)
        out = self.pwconv2(out)
        out = self.gamma[None, :, None, None] * out
        return identidad + out


class ConvNeXtBlockIntermedia(ConvNeXtBlockEntrada):
    """Igual al bloque de entrada, pero con expansion x4 (la proporcion
    estandar del paper original de ConvNeXt en vez de la version reducida)."""
    def __init__(self, c):
        super().__init__(c, expansion=4)


class MobileNetBlockEntrada(nn.Module):
    """Estilo MobileNetV1: convolucion separable en profundidad (depthwise +
    pointwise), sin expansion de canales. El bloque mas barato del catalogo
    junto con Inception."""
    def __init__(self, c):
        super().__init__()
        self.dw = nn.Conv2d(c, c, 3, padding=1, groups=c)
        self.bn1 = nn.BatchNorm2d(c)
        self.pw = nn.Conv2d(c, c, 1)
        self.bn2 = nn.BatchNorm2d(c)
        self.act = nn.ReLU6(inplace=True)

    def forward(self, x):
        x = self.act(self.bn1(self.dw(x)))
        x = self.act(self.bn2(self.pw(x)))
        return x


class MobileNetBlockIntermedia(nn.Module):
    """Estilo MobileNetV2: residual invertido con expansion x6 -- la
    innovacion clave de V2 sobre V1: expandir canales antes de la convolucion
    depthwise y comprimir despues, con conexion de salto."""
    def __init__(self, c, expansion=6):
        super().__init__()
        hidden = c * expansion
        self.expand = nn.Sequential(nn.Conv2d(c, hidden, 1), nn.BatchNorm2d(hidden), nn.ReLU6(inplace=True))
        self.dw = nn.Sequential(
            nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden),
            nn.BatchNorm2d(hidden), nn.ReLU6(inplace=True))
        self.project = nn.Sequential(nn.Conv2d(hidden, c, 1), nn.BatchNorm2d(c))

    def forward(self, x):
        identidad = x
        out = self.expand(x)
        out = self.dw(out)
        out = self.project(out)
        return out + identidad


class BloqueConvCustom(nn.Module):
    """Bloque libre: una convolucion+BN+ReLU con el kernel que el grupo elija.
    Cuesta un valor fijo (5 pts) sin importar el tamano exacto del kernel:
    representa una decision de diseno libre, no una familia de arquitectura
    medida en parametros reales como las demas."""
    def __init__(self, c, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv2d(c, c, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(c), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.net(x)


class ChannelAdapter(nn.Module):
    """Conv 1x1 que normaliza el numero de canales de vuelta al ancho base.
    Necesario despues de cualquier bloque que cambie el numero de canales
    (DenseNet, Inception intermedio) para poder seguir encadenando bloques
    sin errores de forma."""
    def __init__(self, in_c, out_c):
        super().__init__()
        if in_c == out_c:
            self.proj = nn.Identity()
        else:
            self.proj = nn.Sequential(
                nn.Conv2d(in_c, out_c, 1), nn.BatchNorm2d(out_c), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.proj(x)


# ---------------------------------------------------------------------------
# Presupuesto de puntos: costos derivados de parametros reales medidos con
# torch (formula Conv2d = out*(in*kh*kw+1) + BN = 2*canales, validada contra
# medicion real de los 4 bloques originales antes de extenderla al resto del
# catalogo), comprimidos a una escala jugable.
# ---------------------------------------------------------------------------

BLOCK_COSTS = {
    "mobilenet_entrada": 8,
    "inception_entrada": 10,
    "convnext_entrada": 15,
    "inception_intermedia": 20,
    "convnext_intermedia": 25,
    "densenet_entrada": 28,
    "mobilenet_intermedia": 32,
    "vgg_entrada": 38,
    "resnet_entrada": 38,
    "densenet_intermedia": 45,
    "vgg_intermedia": 50,
    "resnet_intermedia": 50,
}

OPTIMIZER_COSTS = {"sgd": 1, "rmsprop": 5, "adam": 4, "adamw": 6}
SCHEDULER_COSTS = {"none": 0, "steplr": 3, "cosine": 5, "plateau": 6}
LOSS_COSTS = {"crossentropy": 1, "label_smoothing": 4, "focal": 8}

COMPONENT_COSTS = {
    "conv_custom": 5,
    "maxpool_extra": 2,
    "avgpool_extra": 2,
    "global_avg_pool": 3,
    "dropout": 2,
    "dropout_espacial": 3,
}

PRESUPUESTO_MIN = 150
PRESUPUESTO_MAX = 200
MAX_EPOCHS = 25


def calcular_presupuesto(bloques, optimizador, scheduler, loss_fn,
                          usar_conv_custom=False, pooling_extra=None,
                          usar_global_avg_pool=True, dropout=None, verbose=True):
    """
    bloques: lista de nombres del catalogo (ver BLOCK_COSTS).
    pooling_extra: None | "maxpool" | "avgpool"
    dropout: None | ("dropout", p) | ("dropout_espacial", p)
    Devuelve (total_puntos, es_valido, detalle).
    """
    for b in bloques:
        if b not in BLOCK_COSTS:
            raise ValueError(f"Bloque desconocido: '{b}'. Opciones validas: {list(BLOCK_COSTS)}")
    if optimizador not in OPTIMIZER_COSTS:
        raise ValueError(f"Optimizador desconocido: '{optimizador}'")
    if scheduler not in SCHEDULER_COSTS:
        raise ValueError(f"Scheduler desconocido: '{scheduler}'")
    if loss_fn not in LOSS_COSTS:
        raise ValueError(f"Funcion de perdida desconocida: '{loss_fn}'")
    if pooling_extra not in (None, "maxpool", "avgpool"):
        raise ValueError(f"pooling_extra debe ser None, 'maxpool' o 'avgpool', recibido: {pooling_extra}")
    if dropout is not None and dropout[0] not in ("dropout", "dropout_espacial"):
        raise ValueError(f"dropout[0] debe ser 'dropout' o 'dropout_espacial', recibido: {dropout[0]}")

    costo_bloques = sum(BLOCK_COSTS[b] for b in bloques)
    costo_opt = OPTIMIZER_COSTS[optimizador]
    costo_sched = SCHEDULER_COSTS[scheduler]
    costo_loss = LOSS_COSTS[loss_fn]

    costo_componentes = 0
    if usar_conv_custom:
        costo_componentes += COMPONENT_COSTS["conv_custom"]
    if pooling_extra == "maxpool":
        costo_componentes += COMPONENT_COSTS["maxpool_extra"]
    elif pooling_extra == "avgpool":
        costo_componentes += COMPONENT_COSTS["avgpool_extra"]
    if usar_global_avg_pool:
        costo_componentes += COMPONENT_COSTS["global_avg_pool"]
    if dropout is not None:
        costo_componentes += COMPONENT_COSTS[dropout[0]]

    total = costo_bloques + costo_opt + costo_sched + costo_loss + costo_componentes
    es_valido = PRESUPUESTO_MIN <= total <= PRESUPUESTO_MAX

    if verbose:
        print(f"Bloques:      {bloques} -> {costo_bloques} pts")
        print(f"Optimizador:  {optimizador} -> {costo_opt} pts")
        print(f"Scheduler:    {scheduler} -> {costo_sched} pts")
        print(f"Loss:         {loss_fn} -> {costo_loss} pts")
        print(f"Componentes:  conv_custom={usar_conv_custom}, pooling_extra={pooling_extra}, "
              f"global_avg_pool={usar_global_avg_pool}, dropout={dropout} -> {costo_componentes} pts")
        print(f"TOTAL: {total} pts (rango permitido: {PRESUPUESTO_MIN}-{PRESUPUESTO_MAX})")
        print("VALIDO" if es_valido else "FUERA DE RANGO - no calificable, ajusten la arquitectura")

    return total, es_valido, {
        "costo_bloques": costo_bloques, "costo_optimizador": costo_opt,
        "costo_scheduler": costo_sched, "costo_loss": costo_loss,
        "costo_componentes": costo_componentes, "total": total,
    }


# ---------------------------------------------------------------------------
# Arquitectura ensamblable
# ---------------------------------------------------------------------------

def _construir_bloque(nombre, c, growth_k=32):
    mapa = {
        "vgg_entrada": (VGGBlockEntrada, c),
        "vgg_intermedia": (VGGBlockIntermedia, c),
        "resnet_entrada": (ResNetBlockEntrada, c),
        "resnet_intermedia": (ResNetBlockIntermedia, c),
        "inception_entrada": (InceptionBlockEntrada, c),
        "inception_intermedia": (InceptionBlockIntermedia, 2 * c),
        "convnext_entrada": (ConvNeXtBlockEntrada, c),
        "convnext_intermedia": (ConvNeXtBlockIntermedia, c),
        "mobilenet_entrada": (MobileNetBlockEntrada, c),
        "mobilenet_intermedia": (MobileNetBlockIntermedia, c),
    }
    if nombre == "densenet_entrada":
        return DenseNetBlockEntrada(c, growth_k), c + growth_k
    if nombre == "densenet_intermedia":
        return DenseNetBlockIntermedia(c, growth_k), c + 2 * growth_k
    if nombre not in mapa:
        raise ValueError(f"Bloque desconocido: '{nombre}'. Opciones validas: {list(BLOCK_COSTS)}")
    clase, canales_salida = mapa[nombre]
    return clase(c), canales_salida


class ChimeraNet(nn.Module):
    """Arquitectura hibrida ensamblable a partir del catalogo de 12 bloques,
    mas componentes opcionales (conv custom, pooling extra, global average
    pooling, dropout / dropout espacial). Reconstruible desde config_dict()
    para el script de calificacion offline."""

    def __init__(self, secuencia_bloques, num_classes, base_channels=64,
                 in_channels=3, usar_conv_custom=False, conv_custom_kernel=3,
                 pooling_extra=None, usar_global_avg_pool=True, dropout=None):
        super().__init__()
        self.secuencia_bloques = list(secuencia_bloques)
        self.num_classes = num_classes
        self.base_channels = base_channels
        self.in_channels = in_channels
        self.usar_conv_custom = usar_conv_custom
        self.conv_custom_kernel = conv_custom_kernel
        self.pooling_extra = pooling_extra
        self.usar_global_avg_pool = usar_global_avg_pool
        self.dropout = dropout

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1),
            nn.BatchNorm2d(base_channels), nn.ReLU(inplace=True))

        capas = []
        if usar_conv_custom:
            capas.append(BloqueConvCustom(base_channels, kernel_size=conv_custom_kernel))
        if pooling_extra == "maxpool":
            capas.append(nn.MaxPool2d(2))
        elif pooling_extra == "avgpool":
            capas.append(nn.AvgPool2d(2))

        for i, nombre in enumerate(self.secuencia_bloques):
            bloque, canales_salida = _construir_bloque(nombre, base_channels)
            capas.append(bloque)
            if canales_salida != base_channels:
                capas.append(ChannelAdapter(canales_salida, base_channels))
            if (i + 1) % 2 == 0:
                capas.append(nn.MaxPool2d(2))

        if dropout is not None and dropout[0] == "dropout_espacial":
            capas.append(nn.Dropout2d(dropout[1]))

        self.features = nn.Sequential(*capas)

        head_layers = []
        if usar_global_avg_pool:
            head_layers.append(nn.AdaptiveAvgPool2d(1))
        head_layers.append(nn.Flatten())
        if dropout is not None and dropout[0] == "dropout":
            head_layers.append(nn.Dropout(dropout[1]))
        if usar_global_avg_pool:
            head_layers.append(nn.Linear(base_channels, num_classes))
        else:
            # Sin global average pooling: se aplana el mapa de caracteristicas
            # completo. LazyLinear infiere el numero de entradas en el primer
            # forward -- por eso el notebook hace un forward de verificacion
            # ANTES de crear el optimizador.
            head_layers.append(nn.LazyLinear(num_classes))
        self.head = nn.Sequential(*head_layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.features(x)
        return self.head(x)

    def config_dict(self):
        """Serializa la configuracion necesaria para reconstruir esta misma
        arquitectura desde cero (usado en el checkpoint firmado)."""
        return {
            "secuencia_bloques": self.secuencia_bloques,
            "num_classes": self.num_classes,
            "base_channels": self.base_channels,
            "in_channels": self.in_channels,
            "usar_conv_custom": self.usar_conv_custom,
            "conv_custom_kernel": self.conv_custom_kernel,
            "pooling_extra": self.pooling_extra,
            "usar_global_avg_pool": self.usar_global_avg_pool,
            "dropout": self.dropout,
        }

    @classmethod
    def from_config(cls, config):
        """Reconstruye una ChimeraNet a partir de un config_dict() guardado."""
        return cls(**config)


# ---------------------------------------------------------------------------
# Focal Loss
# ---------------------------------------------------------------------------

class FocalLoss(nn.Module):
    """Focal Loss (Lin et al., 2017): reduce el peso de ejemplos ya bien
    clasificados para que el entrenamiento se concentre en los dificiles."""
    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(reduction="none")

    def forward(self, logits, target):
        perdida_ce = self.ce(logits, target)
        pt = torch.exp(-perdida_ce)
        return (((1 - pt) ** self.gamma) * perdida_ce).mean()


CRITERIONS = {
    "crossentropy": lambda: nn.CrossEntropyLoss(),
    "label_smoothing": lambda: nn.CrossEntropyLoss(label_smoothing=0.1),
    "focal": lambda: FocalLoss(gamma=2.0),
}

OPTIMIZERS = {
    "sgd": lambda params: optim.SGD(params, lr=0.01, momentum=0.9),
    "rmsprop": lambda params: optim.RMSprop(params, lr=1e-3),
    "adam": lambda params: optim.Adam(params, lr=1e-3),
    "adamw": lambda params: optim.AdamW(params, lr=1e-3, weight_decay=1e-4),
}


def construir_scheduler(nombre, optimizer, max_epochs=MAX_EPOCHS):
    if nombre == "none":
        return None
    if nombre == "steplr":
        return optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.5)
    if nombre == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
    if nombre == "plateau":
        return optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=3, factor=0.5)
    raise ValueError(f"Scheduler desconocido: {nombre}")


# ---------------------------------------------------------------------------
# Entrenamiento y evaluacion
# ---------------------------------------------------------------------------

def evaluar_f1(model, data_loader, device):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in data_loader:
            x = x.to(device)
            preds = model(x).argmax(dim=1).cpu().numpy()
            y_pred.extend(preds)
            y_true.extend(y.numpy())
    return f1_score(y_true, y_pred, average="macro"), y_true, y_pred


def entrenar_modelo(model, train_loader, val_loader, optimizer, criterion,
                     device, max_epochs, scheduler=None, nombre="modelo",
                     modo_backbone_congelado=False):
    """Entrena hasta max_epochs. Si modo_backbone_congelado=True, aplica el
    patron correcto de BatchNorm para transfer learning con backbone congelado:
    todo el modelo en eval(), solo model.head en train()."""
    if max_epochs > MAX_EPOCHS:
        raise ValueError(f"El reto permite maximo {MAX_EPOCHS} epocas; recibiste max_epochs={max_epochs}.")

    historial = {"train_loss": [], "val_f1": []}
    tiempo_inicio = time.time()

    for epoca in range(1, max_epochs + 1):
        if modo_backbone_congelado:
            model.eval()
            model.head.train()
        else:
            model.train()

        perdida_acumulada, n_batches = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            perdida_acumulada += loss.item()
            n_batches += 1

        perdida_promedio = perdida_acumulada / n_batches
        val_f1, _, _ = evaluar_f1(model, val_loader, device)

        if scheduler is not None:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_f1)
            else:
                scheduler.step()

        historial["train_loss"].append(perdida_promedio)
        historial["val_f1"].append(val_f1)
        print(f"[{nombre}] epoca {epoca:2d}/{max_epochs}  loss_train={perdida_promedio:.4f}  f1_val={val_f1:.4f}")

    duracion = time.time() - tiempo_inicio
    print(f"[{nombre}] entrenamiento terminado en {duracion/60:.1f} minutos")
    return historial, tiempo_inicio


F1_MINIMO_ACEPTABLE = 0.55
F1_META = 0.87


def nota_graduada(f1, minimo=F1_MINIMO_ACEPTABLE, meta=F1_META):
    if f1 <= minimo:
        return 0.0
    if f1 >= meta:
        return 1.0
    return (f1 - minimo) / (meta - minimo)


# ---------------------------------------------------------------------------
# Firma anti-fraude
# ---------------------------------------------------------------------------

def calcular_fingerprint(model, batch_fijo, device):
    """Hash reproducible de las predicciones del modelo sobre un batch fijo
    (siempre el mismo, tomado sin shuffle). Cualquier cambio en los pesos --
    reentrenar, cargar otro checkpoint, editar a mano cambia este hash,
    porque se deriva directamente de los parametros finales del modelo."""
    model.eval()
    with torch.no_grad():
        logits = model(batch_fijo.to(device))
        vector = logits.round(decimals=4).cpu().numpy().tobytes()
    return hashlib.sha256(vector).hexdigest()


def guardar_checkpoint_firmado(model, ruta, historial, epocas_entrenadas,
                                codigo_grupo, tiempo_inicio, batch_fijo, device,
                                componente="modelo"):
    """Guarda un checkpoint con metadata verificable: config de arquitectura
    (para reconstruirla exacta), epocas realmente entrenadas, timestamps,
    duracion real, curva de F1 completa por epoca, y un fingerprint que ata
    el archivo a los pesos finales exactos."""
    tiempo_fin = time.time()
    fingerprint = calcular_fingerprint(model, batch_fijo, device)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "config_arquitectura": model.config_dict(),
        "componente": componente,
        "codigo_grupo": codigo_grupo,
        "epocas_entrenadas": epocas_entrenadas,
        "epocas_maximas_permitidas": MAX_EPOCHS,
        "timestamp_inicio": datetime.fromtimestamp(tiempo_inicio).isoformat(),
        "timestamp_fin": datetime.fromtimestamp(tiempo_fin).isoformat(),
        "duracion_segundos": round(tiempo_fin - tiempo_inicio, 1),
        "historial_val_f1": historial["val_f1"],
        "fingerprint_sha256": fingerprint,
    }
    torch.save(checkpoint, ruta)
    print(f"Checkpoint guardado en {ruta}")
    print(f"  epocas_entrenadas={epocas_entrenadas}  duracion={checkpoint['duracion_segundos']:.1f}s  "
          f"fingerprint={fingerprint[:16]}...")
    return checkpoint


def verificar_checkpoint(ruta, batch_fijo, device):
    """Carga un checkpoint, reconstruye la arquitectura exacta desde su
    config, y confirma que el fingerprint guardado coincide con el fingerprint
    recalculado sobre los pesos cargados. Uso: script de calificacion offline."""
    checkpoint = torch.load(ruta, map_location=device, weights_only=False)
    modelo = ChimeraNet.from_config(checkpoint["config_arquitectura"]).to(device)
    modelo.load_state_dict(checkpoint["model_state_dict"])
    fingerprint_recalculado = calcular_fingerprint(modelo, batch_fijo, device)
    coincide = fingerprint_recalculado == checkpoint["fingerprint_sha256"]
    return modelo, checkpoint, coincide
