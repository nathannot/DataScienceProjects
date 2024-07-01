---
layout: default
---

# Accenture Job Tasks

[Back to Projects](/)

{% for file in site.static_files %}
  {% if file.path contains '/Accenture Job Tasks/' %}
    [{{ file.name }}]({{ file.path }})
  {% endif %}
{% endfor %}
