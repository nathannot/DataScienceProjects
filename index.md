# DataScienceProjects

Collection of my data science projects

## Projects

{% for item in site.data.projects %}
- [{{ item.name }}]({{ item.name | replace: ' ', '_' }})
{% endfor %}
