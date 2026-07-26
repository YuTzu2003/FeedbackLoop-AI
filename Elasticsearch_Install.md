# Elasticsearch 9.4.4 + Kibana 9.4.4 Docker Install

## 2. Docker Compose 設定
```yaml
services:
  elasticsearch:
    image: docker.elastic.co/elasticsearch/elasticsearch:9.4.4
    container_name: elasticsearch
    environment:
      discovery.type: single-node
      xpack.security.enabled: "true"
      xpack.security.http.ssl.enabled: "false"
      ELASTIC_PASSWORD: "nfu123@@@"
      ES_JAVA_OPTS: "-Xms1g -Xmx1g"
    ports:
      - "9200:9200"
    volumes:
      - es_data:/usr/share/elasticsearch/data
    restart: unless-stopped
  kibana:
    image: docker.elastic.co/kibana/kibana:9.4.4
    container_name: kibana
    environment:
      ELASTICSEARCH_HOSTS: "http://elasticsearch:9200"
      ELASTICSEARCH_USERNAME: "kibana_system"
      ELASTICSEARCH_PASSWORD: "Kibana123456"
    ports:
      - "5601:5601"
    depends_on:
      - elasticsearch
    restart: unless-stopped
volumes:
  es_data:
```

> `ELASTIC_PASSWORD` 是 Elasticsearch 內建管理員 `elastic` 的密碼。  
> `ELASTICSEARCH_PASSWORD` 是 Kibana 背景服務帳號 `kibana_system` 的密碼。

---

## 3. 先啟動 Elasticsearch

```powershell
docker compose up -d elasticsearch
```

查看狀態：
```powershell
docker compose ps -a
```

查看日誌：
```powershell
docker logs -f elasticsearch
```

測試：
```powershell
curl.exe -u "elastic:nfu123@@@" http://localhost:9200
```

---
## 4. 設定 `kibana_system` 密碼
```powershell
docker exec -it elasticsearch /usr/share/elasticsearch/bin/elasticsearch-reset-password -u kibana_system -i
```

輸入：
```text
Kibana123456
```

此密碼必須和Compose中的密碼完全一致：
```yaml
ELASTICSEARCH_PASSWORD: "Kibana123456"
```

---
## 5. 啟動或重建 Kibana
```powershell
docker compose up -d --force-recreate kibana
```

查看日誌：
```powershell
docker logs -f kibana
```

成功後不應再看到：
```text
missing authentication credentials
```

---

## 6. 開啟 Kibana
```text
http://localhost:5601
```

登入：
```text
帳號：elastic
密碼：nfu123@@@
```

---

## 7. 常用指令
啟動全部服務：
```powershell
docker compose up -d
```

查看容器：
```powershell
docker compose ps
```

停止服務但保留資料：
```powershell
docker compose down
```

停止並刪除資料：
```powershell
docker compose down -v
```

> `docker compose down -v` 會刪除 Elasticsearch 索引與 Docker Volume。

---

## 8. 常見錯誤

### Kibana 顯示 `missing authentication credentials`

確認 Compose 中有：

```yaml
ELASTICSEARCH_USERNAME: "kibana_system"
ELASTICSEARCH_PASSWORD: "Kibana123456"
```

並重新設定密碼後重建 Kibana：

```powershell
docker compose up -d --force-recreate kibana
```

### Elasticsearch 9200 沒有圖形介面
```text
http://localhost:9200
```

REST API；圖形介面是：
```text
http://localhost:5601
```
---

## 9. Kibana Dev Tools 測試

```text
Dev Tools → Console
```

查看索引：
```http
GET /_cat/indices?v
```

建立索引：
```http
PUT /test_index
```

寫入資料：
```http
POST /test_index/_doc
{
  "name": "測試文件",
  "content": "這是一筆 Elasticsearch 測試資料"
}
```

查詢：
```http
GET /test_index/_search
{
  "query": {
    "match": {
      "content": "測試"
    }
  }
}
```