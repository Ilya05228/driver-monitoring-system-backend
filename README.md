# driver-monitoring-system-backend
Ядро системы контроля усталости водителя за рулем

**Мониторит водителя:**  
- Глаза закрыты  
- Голова наклонена  
- Голова вышла из зоны  

**Звук:** `beep_1.mp3` при сне  

## Запуск через **uv**

```bash
uv run main
```

## Конфиг `uv`

```toml
[project.scripts]
main = "driver_monitoring_system_backend.main:main"
```

## Условие сна

```python
eyes_closed OR (≥2 из 3: глаза + наклон + зона)
```

**Готово.**  
Ctrl+C — выход.


# Полезные ссылки
https://habr.com/ru/articles/703496/