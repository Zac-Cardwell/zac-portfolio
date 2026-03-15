import type { Post, WikilinkMatch } from "@/types";
import { visit } from "unist-util-visit";

// Global posts cache for build-time wikilink resolution
let globalPostsCache: any[] = [];

// Function to set the global posts cache
export function setGlobalPostsCache(posts: any[]) {
  globalPostsCache = posts;
}

// Function to get the global posts cache
export function getGlobalPostsCache(): any[] {
  return globalPostsCache;
}

// Function to populate the global posts cache (called from layouts)
export function populateGlobalPostsCache(posts: any[]) {
  globalPostsCache = posts;
}

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

// Utility functions for content-aware URL processing
function isFolderBasedContent(
  collection: "posts" | "pages",
  slug: string,
  allContent: any[]
): boolean {
  const content = allContent.find((item) => item.id === slug);
  return content ? content.id.endsWith("/index") : false;
}

function shouldRemoveIndexFromUrl(
  url: string,
  allPosts: any[],
  allPages: any[]
): boolean {
  // Determine collection type from URL
  if (url.startsWith("/posts/")) {
    const slug = url.replace("/posts/", "").split("#")[0]; // Remove anchor
    return isFolderBasedContent("posts", slug, allPosts);
  } else if (url.startsWith("/pages/")) {
    const slug = url.replace("/pages/", "").split("#")[0]; // Remove anchor
    return isFolderBasedContent("pages", slug, allPages);
  }
  return false; // Don't remove /index for other URLs
}

// Create slug from title for wikilink resolution
function createSlugFromTitle(title: string): string {
  return title
    .toLowerCase()
    .replace(/[^a-z0-9\s-]/g, "")
    .replace(/\s+/g, "-")
    .replace(/-+/g, "-")
    .replace(/^-+|-+$/g, "");
}

// Decode URL-encoded anchor text (for Obsidian compatibility)
function decodeAnchorText(encodedText: string): string {
  try {
    return decodeURIComponent(encodedText);
  } catch (error) {
    return encodedText;
  }
}

// Create anchor slug from text (for heading anchors)
function createAnchorSlug(text: string): string {
  return text
    .toLowerCase()
    .replace(/[^a-z0-9\s-]/g, "")
    .replace(/\s+/g, "-")
    .replace(/-+/g, "-")
    .replace(/^-+|-+$/g, "");
}

// Parse link with potential anchor fragment
function parseLinkWithAnchor(linkText: string): {
  link: string;
  anchor: string | null;
} {
  const anchorIndex = linkText.indexOf("#");
  if (anchorIndex === -1) {
    return { link: linkText, anchor: null };
  }

  const link = linkText.substring(0, anchorIndex);
  const anchor = linkText.substring(anchorIndex + 1);
  const decodedAnchor = anchor ? decodeAnchorText(anchor) : null;

  return { link, anchor: decodedAnchor };
}

// Helper function to check if a node is inside a code block
function isInsideCodeBlock(parent: any, tree: any): boolean {
  if (!parent) return false;
  if (parent.type === "inlineCode" || parent.type === "code") {
    return true;
  }
  let currentNode = parent;
  while (currentNode) {
    if (currentNode.type === "inlineCode" || currentNode.type === "code") {
      return true;
    }
    currentNode = currentNode.parent;
  }
  return false;
}

// Helper function to check if a wikilink is inside backticks in raw content
function isWikilinkInCode(content: string, wikilinkIndex: number): boolean {
  const codeBlockRegex = /```[\s\S]*?```/g;
  let codeBlockMatch;

  while ((codeBlockMatch = codeBlockRegex.exec(content)) !== null) {
    const codeBlockStart = codeBlockMatch.index;
    const codeBlockEnd = codeBlockMatch.index + codeBlockMatch[0].length;
    if (wikilinkIndex >= codeBlockStart && wikilinkIndex < codeBlockEnd) {
      return true;
    }
  }

  const backtickRegex = /`([^`]*)`/g;
  let match;
  while ((match = backtickRegex.exec(content)) !== null) {
    const codeStart = match.index;
    const codeEnd = match.index + match[0].length;
    if (wikilinkIndex >= codeStart && wikilinkIndex < codeEnd) {
      return true;
    }
  }

  return false;
}

// Helper function to check if a URL is an internal link
function isInternalLink(url: string): boolean {
  url = url.trim();
  if (url.startsWith("http://") || url.startsWith("https://")) return false;
  if (url.startsWith("mailto:")) return false;
  if (url.startsWith("#")) return false;

  const { link } = parseLinkWithAnchor(url);

  const isInternal =
    link.endsWith(".md") ||
    link.startsWith("/posts/") ||
    link.startsWith("/pages/") ||
    link.startsWith("/projects/") ||
    link.startsWith("/docs/") ||
    link.startsWith("/special/") ||
    link.startsWith("posts/") ||
    link.startsWith("pages/") ||
    link.startsWith("projects/") ||
    link.startsWith("docs/") ||
    link.startsWith("special/") ||
    !link.includes("/");

  return isInternal;
}

// Helper function to map relative URLs to their actual site URLs
// NOTE: Does NOT prepend base — base is added by callers after mapping.
function mapRelativeUrlToSiteUrl(url: string): string {
  if (url === "/index/" || url === "/index") return "/";

  if (url.startsWith("/special/")) {
    const specialPath = url.replace("/special/", "");
    if (specialPath === "home") return "/";
    if (specialPath === "404") return "/404";
    if (specialPath === "projects") return "/projects";
    if (specialPath === "docs") return "/docs";
    return `/${specialPath}`;
  }

  if (url.startsWith("/pages/")) {
    return `/${url.replace("/pages/", "")}`;
  }

  if (url.startsWith("special/")) {
    const specialPath = url.replace("special/", "");
    if (specialPath === "home") return "/";
    if (specialPath === "404") return "/404";
    if (specialPath === "projects") return "/projects";
    if (specialPath === "docs") return "/docs";
    return `/${specialPath}`;
  }

  if (url.startsWith("pages/")) {
    return `/${url.replace("pages/", "")}`;
  }

  return url;
}

// Helper to prepend base to a root-relative URL (handles "/" correctly)
function withBase(base: string, url: string): string {
  if (!base) return url;
  // url is something like "/posts/foo" or "/" or "/404"
  if (url === "/") return base + "/";
  return base + url;
}

// Helper function to extract link text and anchor from URL for internal links
function extractLinkTextFromUrlWithAnchor(
  url: string,
  allPosts: any[] = [],
  allPages: any[] = []
): { linkText: string | null; anchor: string | null } {
  url = url.trim();
  const { link, anchor } = parseLinkWithAnchor(url);

  if (link.startsWith("posts/")) {
    let linkText = link.replace("posts/", "").replace(/\.md$/, "");
    if (linkText.endsWith("/index") && linkText.split("/").length === 2) {
      linkText = linkText.replace("/index", "");
    }
    return { linkText, anchor };
  }

  if (link.startsWith("/posts/")) {
    let linkText = link.replace("/posts/", "").replace(/\.md$/, "");
    if (linkText.endsWith("/index") && linkText.split("/").length === 2) {
      linkText = linkText.replace("/index", "");
    }
    return { linkText, anchor };
  }

  if (link.startsWith("special/")) {
    const specialPath = link.replace("special/", "").replace(/\.md$/, "");
    if (specialPath === "home") return { linkText: "homepage", anchor };
    if (specialPath === "404") return { linkText: "404", anchor };
    return { linkText: specialPath, anchor };
  }

  if (link.startsWith("/special/")) {
    const specialPath = link.replace("/special/", "");
    if (specialPath === "home") return { linkText: "homepage", anchor };
    if (specialPath === "404") return { linkText: "404", anchor };
    return { linkText: specialPath, anchor };
  }

  if (link.startsWith("/pages/")) {
    let linkText = link.replace("/pages/", "").replace(/\.md$/, "");
    if (linkText.endsWith("/index")) linkText = linkText.replace("/index", "");
    return { linkText: linkText === "" ? "homepage" : linkText, anchor };
  }

  if (link.startsWith("pages/")) {
    let linkText = link.replace("pages/", "").replace(/\.md$/, "");
    if (linkText.endsWith("/index")) linkText = linkText.replace("/index", "");
    return { linkText: linkText === "" ? "homepage" : linkText, anchor };
  }

  if (link.endsWith(".md")) {
    let linkText = link.replace(/\.md$/, "");
    if (linkText.endsWith("/index") && linkText.split("/").length === 2) {
      linkText = linkText.replace("/index", "");
    }
    return { linkText, anchor };
  }

  if (!link.includes("/")) {
    return { linkText: link, anchor };
  }

  return { linkText: null, anchor: null };
}

// ============================================================================
// WIKILINKS (OBSIDIAN-STYLE) - POSTS ONLY
// ============================================================================

export function remarkWikilinks(base: string = '') {
  return function transformer(tree: any, file: any) {
    const nodesToReplace: Array<{
      parent: any;
      index: number;
      newChildren: any[];
    }> = [];

    visit(tree, "text", (node: any, index: any, parent: any) => {
      if (!node.value || typeof node.value !== "string") return;
      if (isInsideCodeBlock(parent, tree)) return;

      const wikilinkRegex = /!?\[\[([^\]]+)\]\]/g;
      let match;
      const newChildren: any[] = [];
      let lastIndex = 0;
      let hasWikilinks = false;

      while ((match = wikilinkRegex.exec(node.value)) !== null) {
        hasWikilinks = true;
        const [fullMatch, content] = match;
        const isImageWikilink = fullMatch.startsWith("!");
        const [link, displayText] = content.includes("|")
          ? content.split("|", 2)
          : [content, null];

        if (match.index > lastIndex) {
          newChildren.push({
            type: "text",
            value: node.value.slice(lastIndex, match.index),
          });
        }

        const linkText = link.trim();
        const finalDisplayText = displayText ? displayText.trim() : linkText;

        if (isImageWikilink) {
          const imagePath = linkText;
          const altText = displayText || "";
          newChildren.push({
            type: "image",
            url: imagePath,
            alt: altText,
            title: null,
            data: {
              hName: "img",
              hProperties: { src: imagePath, alt: altText },
            },
          });
        } else {
          const { link, anchor } = parseLinkWithAnchor(linkText);
          const isSamePageAnchor = linkText.startsWith("#") || link === "";

          let url: string;
          let wikilinkData: string;

          if (isSamePageAnchor) {
            const anchorText = linkText.startsWith("#")
              ? linkText.substring(1)
              : linkText;
            const decodedAnchor = decodeAnchorText(anchorText);
            const anchorSlug = createAnchorSlug(decodedAnchor);
            url = `#${anchorSlug}`;
            wikilinkData = "";
          } else if (link.startsWith("posts/")) {
            const postPath = link.replace("posts/", "");
            const cleanPath =
              postPath.endsWith("/index") && postPath.split("/").length === 2
                ? postPath.replace("/index", "")
                : postPath;
            url = withBase(base, `/posts/${cleanPath}`);
            wikilinkData = cleanPath;
          } else if (link.includes("/")) {
            if (link.endsWith("/index") && link.split("/").length === 2) {
              const folderName = link.replace("/index", "");
              url = withBase(base, `/posts/${folderName}`);
              wikilinkData = folderName;
            } else {
              return;
            }
          } else {
            const slugifiedLink = createSlugFromTitle(link);
            url = withBase(base, `/posts/${slugifiedLink}`);
            wikilinkData = link.trim();
          }

          if (anchor && !isSamePageAnchor) {
            const anchorSlug = createAnchorSlug(anchor);
            if (!url.includes('#')) {
              url += `#${anchorSlug}`;
            }
          }

          newChildren.push({
            type: "link",
            url: url,
            title: null,
            data: {
              hName: "a",
              hProperties: {
                className: ["wikilink"],
                "data-wikilink": wikilinkData,
                "data-display-override": displayText,
              },
            },
            children: [
              {
                type: "text",
                value: displayText || (isSamePageAnchor ? linkText.replace(/^#/, "") : link.trim()),
              },
            ],
          });
        }

        lastIndex = wikilinkRegex.lastIndex;
      }

      if (lastIndex < node.value.length) {
        newChildren.push({
          type: "text",
          value: node.value.slice(lastIndex),
        });
      }

      if (hasWikilinks && parent && parent.children) {
        nodesToReplace.push({ parent, index, newChildren });
      }
    });

    nodesToReplace.reverse().forEach(({ parent, index, newChildren }) => {
      if (parent && parent.children && Array.isArray(parent.children)) {
        parent.children.splice(index, 1, ...newChildren);
      }
    });
  };
}

// Extract wikilinks from content (Obsidian-style)
export function extractWikilinks(content: string): WikilinkMatch[] {
  const matches: WikilinkMatch[] = [];
  const wikilinkRegex = /!?\[\[([^\]]+)\]\]/g;
  let wikilinkMatch;

  while ((wikilinkMatch = wikilinkRegex.exec(content)) !== null) {
    const [fullMatch, linkContent] = wikilinkMatch;
    const isImageWikilink = fullMatch.startsWith("!");
    if (isWikilinkInCode(content, wikilinkMatch.index)) continue;

    if (!isImageWikilink) {
      const [link, displayText] = linkContent.includes("|")
        ? linkContent.split("|", 2)
        : [linkContent, linkContent];

      const { link: baseLink } = parseLinkWithAnchor(link.trim());
      if (link.trim().startsWith("#") || baseLink === "") continue;

      let slug = baseLink;
      if (baseLink.startsWith("posts/")) {
        const postPath = baseLink.replace("posts/", "");
        if (postPath.endsWith("/index") && postPath.split("/").length === 2) {
          slug = postPath.replace("/index", "");
        } else {
          slug = postPath;
        }
      } else if (baseLink.includes("/")) {
        if (baseLink.endsWith("/index") && baseLink.split("/").length === 2) {
          slug = baseLink.replace("/index", "");
        }
      }

      const finalSlug = slug
        .toLowerCase()
        .replace(/[^a-z0-9\s-]/g, "")
        .replace(/\s+/g, "-")
        .replace(/-+/g, "-")
        .replace(/^-+|-+$/g, "");

      matches.push({
        link: baseLink,
        display: displayText.trim(),
        slug: finalSlug,
      });
    }
  }

  return matches;
}

// Resolve wikilink to actual post (posts only)
export function resolveWikilink(posts: Post[], linkText: string): Post | null {
  const targetSlug = createSlugFromTitle(linkText);
  let post = posts.find((p) => p.id === targetSlug);
  if (!post) {
    post = posts.find((p) => createSlugFromTitle(p.data.title) === targetSlug);
  }
  return post || null;
}

// Validate wikilinks in content (posts only)
export function validateWikilinks(
  posts: Post[],
  content: string
): { valid: WikilinkMatch[]; invalid: WikilinkMatch[] } {
  const wikilinks = extractWikilinks(content);
  const valid: WikilinkMatch[] = [];
  const invalid: WikilinkMatch[] = [];

  wikilinks.forEach((wikilink) => {
    const resolved = resolveWikilink(posts, wikilink.link);
    if (resolved) {
      valid.push(wikilink);
    } else {
      invalid.push(wikilink);
    }
  });

  return { valid, invalid };
}

// ============================================================================
// STANDARD MARKDOWN LINKS - ALL CONTENT TYPES
// ============================================================================

export function remarkStandardLinks(base: string = '') {
  return function transformer(tree: any, file: any) {
    visit(tree, "link", (node: any) => {
      if (!node.url) return;

      // Handle same-page anchor links
      if (node.url.startsWith("#") && node.url.length > 1) {
        let anchorText = node.url.substring(1);
        try {
          anchorText = decodeURIComponent(anchorText);
        } catch {
          // use as-is
        }
        const anchorSlug = createAnchorSlug(anchorText);
        const normalizedUrl = `#${anchorSlug}`;
        node.url = normalizedUrl;
        if (!node.data) node.data = {};
        if (!node.data.hProperties) node.data.hProperties = {};
        node.data.hProperties.href = normalizedUrl;
        node.data.hProperties.className = node.data.hProperties.className || [];
        if (!Array.isArray(node.data.hProperties.className)) {
          node.data.hProperties.className = [node.data.hProperties.className];
        }
        if (!node.data.hProperties.className.includes('wikilink')) {
          node.data.hProperties.className.push('wikilink');
        }
        return;
      }

      if (isInternalLink(node.url)) {
        const { linkText, anchor } = extractLinkTextFromUrlWithAnchor(node.url);
        if (linkText) {
          // Handle /pages/ URLs that don't end in .md
          if (
            node.url.startsWith("/pages/") &&
            !node.url.endsWith(".md") &&
            !node.url.includes(".md#")
          ) {
            let mappedUrl = mapRelativeUrlToSiteUrl(node.url);
            if (anchor && !mappedUrl.includes("#")) {
              mappedUrl += `#${createAnchorSlug(anchor)}`;
            }
            node.url = withBase(base, mappedUrl);
          }
          // Convert .md file references to proper URLs
          else if (node.url.endsWith(".md") || node.url.includes(".md#")) {
            let baseUrl = "";

            if (node.url.startsWith("special/")) {
              const specialPath = node.url.replace("special/", "").replace(/\.md.*$/, "");
              if (specialPath === "home") baseUrl = "/";
              else if (specialPath === "404") baseUrl = "/404";
              else baseUrl = `/${specialPath}`;
            } else if (linkText === "homepage") {
              baseUrl = "/";
            } else if (linkText === "404") {
              baseUrl = "/404";
            } else if (node.url.startsWith("posts/")) {
              let postPath = node.url.replace("posts/", "").replace(/\.md.*$/, "");
              if (postPath.endsWith("/index") && postPath.split("/").length === 2) {
                postPath = postPath.replace("/index", "");
              }
              baseUrl = `/posts/${postPath}`;
            } else if (node.url.startsWith("pages/")) {
              baseUrl = mapRelativeUrlToSiteUrl(node.url.replace(/\.md.*$/, ""));
            } else if (node.url.startsWith("/pages/")) {
              baseUrl = mapRelativeUrlToSiteUrl(node.url);
            } else if (node.url.startsWith("/special/")) {
              baseUrl = mapRelativeUrlToSiteUrl(node.url);
            } else if (node.url.startsWith("special/")) {
              baseUrl = mapRelativeUrlToSiteUrl(node.url);
            } else if (node.url.startsWith("projects/")) {
              baseUrl = `/${node.url.replace(/\.md.*$/, "")}`;
              if (baseUrl.endsWith("/index") && baseUrl.split("/").length === 4) {
                baseUrl = baseUrl.replace("/index", "");
              }
            } else if (node.url.startsWith("docs/")) {
              baseUrl = `/${node.url.replace(/\.md.*$/, "")}`;
              if (baseUrl.endsWith("/index") && baseUrl.split("/").length === 4) {
                baseUrl = baseUrl.replace("/index", "");
              }
            } else {
              if (linkText === "special/home") baseUrl = "/";
              else if (linkText === "special/404") baseUrl = "/404";
              else if (linkText.startsWith("special/")) {
                baseUrl = `/${linkText.replace("special/", "")}`;
              } else {
                if (!linkText) {
                  let processedUrl = node.url.replace(/\.md.*$/, "").replace(/\/index$/, "");
                  baseUrl = `/posts/${processedUrl}`;
                } else {
                  let cleanLinkText = linkText.replace(/\/index$/, "");
                  baseUrl = `/posts/${cleanLinkText}`;
                }
              }
            }

            // Remove /index defensively
            baseUrl = baseUrl.replace(/\/index$/, "");
            baseUrl = baseUrl.replace(/\/index#/, "#");

            if (anchor) {
              baseUrl += `#${createAnchorSlug(anchor)}`;
            }

            let finalUrl = baseUrl.replace(/\/index(?=#|$)/g, "");

            // Prepend base (mapRelativeUrlToSiteUrl returns root-relative paths)
            node.url = withBase(base, finalUrl);
          } else {
            // Non-.md URLs
            if (node.url.startsWith("posts/")) {
              let postPath = node.url.replace("posts/", "");
              const pathWithoutAnchor = postPath.split('#')[0];
              const cleanPath = pathWithoutAnchor.replace(/\/index$/, "");
              let mappedUrl = `/posts/${cleanPath}`;
              if (anchor) mappedUrl += `#${createAnchorSlug(anchor)}`;
              node.url = withBase(base, mappedUrl);
            } else {
              let mappedUrl = mapRelativeUrlToSiteUrl(node.url);
              if (anchor && !mappedUrl.includes("#")) {
                mappedUrl += `#${createAnchorSlug(anchor)}`;
              }
              node.url = withBase(base, mappedUrl);
            }
          }

          // Final safety pass: fix any /posts/ URLs that still have /index
          if (node.url && typeof node.url === 'string') {
            if (node.url.includes("/posts/")) {
              let fixedUrl = node.url;
              fixedUrl = fixedUrl.replace(/\/index(?=#|$)/g, "");
              fixedUrl = fixedUrl.replace(/\/index$/g, "");
              if (fixedUrl.endsWith("/index")) fixedUrl = fixedUrl.slice(0, -6);
              node.url = fixedUrl;
            }
          }

          // Add wikilink styling to internal post links
          if (node.url.startsWith(base + "/posts/") || node.url.startsWith("/posts/")) {
            if (!node.data) node.data = {};
            if (!node.data.hProperties) node.data.hProperties = {};
            const existingClasses = node.data.hProperties.className || [];
            node.data.hProperties.className = Array.isArray(existingClasses)
              ? [...existingClasses, "wikilink"]
              : [existingClasses, "wikilink"].filter(Boolean);
          }
        }
      }
    });
  };
}

// Extract standard markdown links from content (all content types)
export function extractStandardLinks(content: string): WikilinkMatch[] {
  const matches: WikilinkMatch[] = [];
  const markdownLinkRegex = /\[([^\]]+)\]\(([^)]+)\)/g;
  let markdownMatch;

  while ((markdownMatch = markdownLinkRegex.exec(content)) !== null) {
    const [fullMatch, displayText, url] = markdownMatch;
    if (isWikilinkInCode(content, markdownMatch.index)) continue;

    if (isInternalLink(url)) {
      const { linkText } = extractLinkTextFromUrlWithAnchor(url);
      if (linkText) {
        const isPostLink =
          linkText.startsWith("posts/") ||
          url.startsWith("/posts/") ||
          url.startsWith("posts/") ||
          url.endsWith(".md") ||
          (!linkText.includes("/") && !url.startsWith("/"));

        if (isPostLink) {
          let slug = linkText;
          if (linkText.startsWith("posts/")) {
            const postPath = linkText.replace("posts/", "");
            if (postPath.endsWith("/index") && postPath.split("/").length === 2) {
              slug = postPath.replace("/index", "");
            } else {
              slug = postPath;
            }
          } else if (linkText.includes("/")) {
            if (linkText.endsWith("/index") && linkText.split("/").length === 2) {
              slug = linkText.replace("/index", "");
            }
          }

          const finalSlug = slug
            .toLowerCase()
            .replace(/[^a-z0-9\s-]/g, "")
            .replace(/\s+/g, "-")
            .replace(/-+/g, "-")
            .replace(/^-+|-+$/g, "");

          matches.push({
            link: linkText,
            display: displayText.trim(),
            slug: finalSlug,
          });
        }
      }
    }
  }

  return matches;
}

// ============================================================================
// COMBINED LINK PROCESSING
// ============================================================================

export function remarkInternalLinks(options: { base?: string } = {}) {
  const base = options.base || '';
  return function transformer(tree: any, file: any) {
    const wikilinkPlugin = remarkWikilinks(base);
    wikilinkPlugin(tree, file);
    const standardLinkPlugin = remarkStandardLinks(base);
    standardLinkPlugin(tree, file);
  };
}

// Extract all internal links (both wikilinks and standard links)
export function extractAllInternalLinks(content: string): WikilinkMatch[] {
  const wikilinks = extractWikilinks(content);
  const standardLinks = extractStandardLinks(content);
  const allLinks = [...wikilinks, ...standardLinks];
  return allLinks.filter(
    (link, index, self) => index === self.findIndex((l) => l.slug === link.slug)
  );
}

// ============================================================================
// LINKED MENTIONS (POSTS ONLY)
// ============================================================================

export function findLinkedMentions(
  posts: Post[],
  targetSlug: string,
  allPosts: any[] = [],
  allPages: any[] = []
) {
  const mentions = posts
    .filter((post) => post.id !== targetSlug)
    .map((post) => {
      if (!post.body) return null;

      const wikilinks = extractWikilinks(post.body);
      const standardLinks = extractStandardLinks(post.body);
      const allLinks = [...wikilinks, ...standardLinks];
      const matchingLinks = allLinks.filter((link) => link.slug === targetSlug);

      if (matchingLinks.length > 0) {
        const originalLinkText = matchingLinks[0].link;
        const excerptResult = createExcerptAroundWikilink(
          post.body,
          originalLinkText,
          allPosts,
          allPages
        );
        return {
          title: post.data.title,
          slug: post.id,
          excerpt: excerptResult,
        };
      }
      return null;
    })
    .filter(Boolean);

  return mentions;
}

// Create excerpt around wikilink for context
function createExcerptAroundWikilink(
  content: string,
  linkText: string,
  allPosts: any[] = [],
  allPages: any[] = []
): string {
  const withoutFrontmatter = content.replace(/^---\n[\s\S]*?\n---\n/, "");
  const wikilinkPattern = `\\[\\[${linkText}[^\\]]*\\]\\]`;
  const wikilinkRegex = new RegExp(wikilinkPattern, "i");

  let match;
  let searchStart = 0;

  while (
    (match = wikilinkRegex.exec(withoutFrontmatter.slice(searchStart))) !== null
  ) {
    const actualIndex = searchStart + match.index!;
    if (!isWikilinkInCode(withoutFrontmatter, actualIndex)) {
      const result = extractExcerptAtPosition(
        withoutFrontmatter,
        actualIndex,
        match[0].length,
        withoutFrontmatter.length
      );
      return result.excerpt;
    }
    searchStart = actualIndex + match[0].length;
    wikilinkRegex.lastIndex = 0;
  }

  const markdownLinkRegex = /\[([^\]]+)\]\(([^)]+)\)/g;
  let markdownMatch;
  markdownLinkRegex.lastIndex = 0;

  while ((markdownMatch = markdownLinkRegex.exec(withoutFrontmatter)) !== null) {
    const [fullMatch, displayText, url] = markdownMatch;
    if (!isWikilinkInCode(withoutFrontmatter, markdownMatch.index)) {
      if (isInternalLink(url)) {
        const { linkText: urlLinkText } = extractLinkTextFromUrlWithAnchor(
          url,
          allPosts,
          allPages
        );
        if (urlLinkText) {
          const normalizedLinkText = linkText.replace(/\/index$/, "");
          const normalizedUrlLinkText = urlLinkText.replace(/\/index$/, "");
          if (
            normalizedUrlLinkText === normalizedLinkText ||
            urlLinkText === linkText
          ) {
            const result = extractExcerptAtPosition(
              withoutFrontmatter,
              markdownMatch.index,
              fullMatch.length,
              withoutFrontmatter.length
            );
            return result.excerpt;
          }
        }
      }
    }
  }

  return "";
}

// Helper function to extract excerpt at a specific position
function extractExcerptAtPosition(
  content: string,
  position: number,
  linkLength: number,
  contentLength: number
): { excerpt: string; isAtStart: boolean; isAtEnd: boolean } {
  const contextLength = 100;
  const minContextLength = 60;
  const maxExcerptLength = 200;

  const isNearEnd = (contentLength - (position + linkLength)) < minContextLength;
  const contextBeforeLink = isNearEnd ? Math.max(contextLength * 2, 250) : contextLength;

  let start = Math.max(0, position - contextBeforeLink);
  let end = Math.min(contentLength, position + linkLength + (isNearEnd ? 0 : contextLength));

  const codeBlockRegex = /```[\s\S]*?```/g;
  let codeBlockMatch;
  codeBlockRegex.lastIndex = 0;

  while ((codeBlockMatch = codeBlockRegex.exec(content)) !== null) {
    const codeBlockStart = codeBlockMatch.index;
    const codeBlockEnd = codeBlockMatch.index + codeBlockMatch[0].length;

    if (position >= codeBlockStart && position < codeBlockEnd) {
      return { excerpt: "", isAtStart: false, isAtEnd: false };
    }
    if (codeBlockStart > start && codeBlockStart < end && codeBlockStart > position) {
      end = codeBlockStart;
    }
    if (codeBlockEnd > start && codeBlockEnd < position && codeBlockStart < start) {
      start = codeBlockEnd;
    }
  }

  const contextBefore = position - start;
  const contextAfter = end - (position + linkLength);

  if (contextBefore < minContextLength && start > 0) {
    const neededBefore = minContextLength - contextBefore;
    const desiredStart = Math.max(0, start - neededBefore);
    let canExtend = true;
    codeBlockRegex.lastIndex = 0;
    while ((codeBlockMatch = codeBlockRegex.exec(content)) !== null) {
      const codeBlockEnd = codeBlockMatch.index + codeBlockMatch[0].length;
      if (codeBlockEnd > desiredStart && codeBlockEnd <= start) {
        canExtend = false;
        break;
      }
    }
    if (canExtend) start = desiredStart;
  }

  if (contextAfter < minContextLength && end < contentLength) {
    const neededAfter = minContextLength - contextAfter;
    const desiredEnd = Math.min(contentLength, end + neededAfter);
    let canExtend = true;
    codeBlockRegex.lastIndex = 0;
    while ((codeBlockMatch = codeBlockRegex.exec(content)) !== null) {
      const codeBlockStart = codeBlockMatch.index;
      if (codeBlockStart >= end && codeBlockStart < desiredEnd) {
        canExtend = false;
        break;
      }
    }
    if (canExtend) end = desiredEnd;
  }

  const matchContent = content.substring(position, Math.min(position + 100, contentLength));
  if (matchContent.startsWith('[[')) {
    const fullWikilinkMatch = content.substring(position).match(/^\[\[([^\]]+)(?:\|([^\]]+))?\]\]/);
    if (fullWikilinkMatch) {
      const actualLinkLength = fullWikilinkMatch[0].length;
      if (actualLinkLength > linkLength) {
        linkLength = actualLinkLength;
        end = Math.min(contentLength, position + linkLength + contextLength);
      }
    }
  }

  const afterPosition = content.substring(position + linkLength);
  const markdownLinkMatch = afterPosition.match(/^[^\]]*\][^\)]*\)/);
  if (markdownLinkMatch && end < position + linkLength + markdownLinkMatch[0].length) {
    end = Math.min(contentLength, position + linkLength + markdownLinkMatch[0].length + 50);
  }

  const wikilinkMatch = afterPosition.match(/^\[\[([^\]]+)(?:\|([^\]]+))?\]\]/);
  if (wikilinkMatch && end < position + linkLength + wikilinkMatch[0].length) {
    end = Math.min(contentLength, position + linkLength + wikilinkMatch[0].length + 50);
  }

  const isAtStart = start === 0;
  const isAtEnd = end >= contentLength;

  let excerpt = content.slice(start, end);

  excerpt = excerpt
    .replace(/\n+/g, " ")
    .replace(/\s+/g, " ")
    .trim()
    .replace(/```[\s\S]*?```/g, " ")
    .replace(/```[\s\S]*$/g, " ")
    .replace(/^[\s\S]*?```/g, " ")
    .replace(/```+/g, " ")
    .replace(/`([^`\n]+)`/g, " ")
    .replace(/\*\*([^*]+?)\*\*/g, "$1")
    .replace(/\*([^*\s][^*]*?[^*\s])\*/g, "$1")
    .replace(/\*([^*\s]+)\*/g, "$1")
    .replace(/_{1,2}([^_]+)_{1,2}/g, "$1")
    .replace(/~~([^~]+)~~/g, "$1")
    .replace(/#{1,6}\s+/g, "")
    .replace(/\s*>\s*\[![\w-]+\]\s*/g, " ")
    .replace(/\s*>\s*/g, " ")
    .replace(/\s*---+\s*/g, " ")
    .replace(/\s*\[![\w-]+\]\s*/g, " ")
    .replace(/^-\s+/gm, "")
    .replace(/^\d+\.\s+/gm, "")
    .replace(/\*\*+/g, "")
    .replace(/\s+/g, " ")
    .trim();

  for (let pass = 0; pass < 5; pass++) {
    excerpt = excerpt
      .replace(/([A-Z][a-z]+):\s*-\s*([A-Z][a-z]+):/g, "")
      .replace(/\b([A-Z][a-z]+):\s*-\s*(?=[A-Z][a-z]+:|$)/g, "")
      .replace(/\b([A-Z][a-z]+(?:\s+[a-z]+)?):\s*$/g, "")
      .replace(/([a-z\s]+)\s+-\s+([A-Z][a-z]+(?:\s+[a-z]+)?):\s*(?=[A-Z]|\[|$)/g, "$1 ")
      .replace(/\s*-\s*(?=[A-Z][a-z]+(?:\s+[a-z]+)?:)/g, " ")
      .replace(/:\s*$/, "")
      .replace(/\s*-\s*$/, "")
      .replace(/\s+/g, " ")
      .trim();
  }

  excerpt = excerpt
    .replace(/```+/g, " ")
    .replace(/\s*```\s*/g, " ")
    .replace(/\b([A-Z][a-z]+(?:\s+[a-z]+)?):\s+(?=[A-Z][a-z]+(?:\s+[a-z]+)?:|$)/g, " ")
    .replace(/\b([A-Z][a-z]+(?:\s+[a-z]+)?):\s*$/g, "")
    .replace(/\s+/g, " ")
    .trim();

  for (let dupPass = 0; dupPass < 2; dupPass++) {
    excerpt = excerpt
      .replace(/\b(\w+)\s+\1\b(?=\s|$)/gi, "$1")
      .replace(/(\[\[[^\]]+\]\])\s+\1(?=\s|$)/g, "$1")
      .replace(/(\[[^\]]+\]\([^\)]+\))\s+\1(?=\s|$)/g, "$1")
      .replace(/\s+/g, " ")
      .trim();
  }

  excerpt = excerpt.replace(/\s+/g, " ").trim();

  const cleanedWords = excerpt.split(/\s+/).filter(w => w.length > 0 && w.match(/[a-zA-Z0-9]/));
  const minCleanedWords = 10;
  const minCleanedLength = 60;

  if ((cleanedWords.length < minCleanedWords || excerpt.length < minCleanedLength) && start > 0) {
    const currentRawLength = end - start;
    const ratio = excerpt.length > 0 ? currentRawLength / excerpt.length : 3;
    const neededCleaned = Math.max(minCleanedLength - excerpt.length, (minCleanedWords - cleanedWords.length) * 8);
    const additionalRaw = Math.ceil(neededCleaned * Math.max(ratio, 2) * 1.5);
    const newStart = Math.max(0, start - additionalRaw);

    let canExtend = true;
    const codeBlockRegex4 = /```[\s\S]*?```/g;
    codeBlockRegex4.lastIndex = 0;
    while ((codeBlockMatch = codeBlockRegex4.exec(content)) !== null) {
      const codeBlockEnd = codeBlockMatch.index + codeBlockMatch[0].length;
      if (codeBlockEnd > newStart && codeBlockEnd <= start) {
        canExtend = false;
        break;
      }
    }

    if (canExtend && newStart < start) {
      start = newStart;
      let newExcerpt = content.slice(start, end);

      newExcerpt = newExcerpt
        .replace(/\n+/g, " ")
        .replace(/\s+/g, " ")
        .trim()
        .replace(/```[\s\S]*?```/g, " ")
        .replace(/```[\s\S]*$/g, " ")
        .replace(/^[\s\S]*?```/g, " ")
        .replace(/```+/g, " ")
        .replace(/`([^`]+)`/g, " ")
        .replace(/\*\*([^*]+?)\*\*/g, "$1")
        .replace(/\*([^*\s][^*]*?[^*\s])\*/g, "$1")
        .replace(/\*([^*\s]+)\*/g, "$1")
        .replace(/_{1,2}([^_]+)_{1,2}/g, "$1")
        .replace(/~~([^~]+)~~/g, "$1")
        .replace(/#{1,6}\s+/g, "")
        .replace(/\s*>\s*\[![\w-]+\]\s*/g, " ")
        .replace(/\s*>\s*/g, " ")
        .replace(/\s*---+\s*/g, " ")
        .replace(/\s*\[![\w-]+\]\s*/g, " ")
        .replace(/^-\s+/gm, "")
        .replace(/^\d+\.\s+/gm, "")
        .replace(/\*\*+/g, "")
        .replace(/\s+/g, " ")
        .trim();

      for (let pass = 0; pass < 5; pass++) {
        newExcerpt = newExcerpt
          .replace(/([A-Z][a-z]+):\s*-\s*([A-Z][a-z]+):/g, "")
          .replace(/\b([A-Z][a-z]+):\s*-\s*/g, "")
          .replace(/\b([A-Z][a-z]+(?:\s+[a-z]+)?):\s*$/g, "")
          .replace(/([a-z\s]+)\s+-\s+([A-Z][a-z]+(?:\s+[a-z]+)?):\s*(?=[A-Z]|\[|$)/g, "$1 ")
          .replace(/\s*-\s*(?=[A-Z][a-z]+(?:\s+[a-z]+)?:)/g, " ")
          .replace(/:\s*$/, "")
          .replace(/\s*-\s*$/, "")
          .replace(/\s+/g, " ")
          .trim();
      }

      newExcerpt = newExcerpt
        .replace(/```+/g, " ")
        .replace(/\s*```\s*/g, " ")
        .replace(/\b([A-Z][a-z]+(?:\s+[a-z]+)?):\s+(?=[A-Z][a-z]+(?:\s+[a-z]+)?:|$)/g, " ")
        .replace(/\b([A-Z][a-z]+(?:\s+[a-z]+)?):\s*$/g, "")
        .replace(/\s+/g, " ")
        .trim();

      excerpt = newExcerpt;
    }
  }

  if (excerpt.length > maxExcerptLength) {
    const truncated = excerpt.substring(0, maxExcerptLength);
    const incompleteWikilinkMatch = truncated.match(/\[\[[^\]]*$/);
    const incompleteMarkdownMatch = truncated.match(/\[[^\]]*\]\([^\)]*$/);
    const afterTruncation = excerpt.substring(maxExcerptLength);
    const completeLinkAfter = afterTruncation.match(/^[^\[]*(\[[^\]]+\]\([^\)]+\)|\[\[[^\]]+\]\])/);

    let truncateAt = maxExcerptLength;

    if (incompleteWikilinkMatch || incompleteMarkdownMatch) {
      if (completeLinkAfter && completeLinkAfter.index !== undefined) {
        const linkEnd = maxExcerptLength + (completeLinkAfter.index || 0) + completeLinkAfter[1].length;
        if (linkEnd <= maxExcerptLength * 1.5) {
          truncateAt = linkEnd;
          const afterLink = excerpt.substring(truncateAt, truncateAt + 30);
          const nextSpace = afterLink.indexOf(' ');
          if (nextSpace > 0) truncateAt = truncateAt + nextSpace;
        } else {
          const incompleteStart = incompleteWikilinkMatch
            ? (incompleteWikilinkMatch.index || 0)
            : (incompleteMarkdownMatch?.index || 0);
          if (incompleteStart > 0) truncateAt = incompleteStart;
        }
      } else {
        const incompleteStart = incompleteWikilinkMatch
          ? (incompleteWikilinkMatch.index || 0)
          : (incompleteMarkdownMatch?.index || 0);
        if (incompleteStart > 0) truncateAt = incompleteStart;
      }
    } else if (completeLinkAfter && completeLinkAfter.index !== undefined && completeLinkAfter.index < 20) {
      const linkEnd = maxExcerptLength + (completeLinkAfter.index || 0) + completeLinkAfter[1].length;
      if (linkEnd <= maxExcerptLength * 1.3) {
        truncateAt = linkEnd;
        const afterLink = excerpt.substring(truncateAt, truncateAt + 30);
        const nextSpace = afterLink.indexOf(' ');
        if (nextSpace > 0) truncateAt = truncateAt + nextSpace;
      }
    }

    const toTruncate = excerpt.substring(0, truncateAt);
    const lastSpace = toTruncate.lastIndexOf(' ');

    if (lastSpace > 10) {
      excerpt = toTruncate.substring(0, lastSpace).trim();
    } else if (lastSpace > 0) {
      excerpt = toTruncate.substring(0, lastSpace).trim();
    } else {
      const lastWhitespace = toTruncate.search(/\s+$/);
      if (lastWhitespace > 0) {
        excerpt = toTruncate.substring(0, lastWhitespace).trim();
      } else {
        excerpt = toTruncate.trim();
      }
    }
  }

  return { excerpt, isAtStart, isAtEnd };
}

// ============================================================================
// HTML PROCESSING
// ============================================================================

export function processWikilinksInHTML(
  posts: Post[],
  html: string,
  allPosts: any[] = [],
  allPages: any[] = []
): string {
  return html;
}

export function processContentAwareWikilinks(
  content: string,
  allPosts: any[],
  allPages: any[]
): string {
  return content;
}

// ============================================================================
// IMAGE PROCESSING
// ============================================================================

function convertToWebP(imagePath: string): string {
  if (!imagePath ||
      imagePath.startsWith("http") ||
      imagePath.toLowerCase().endsWith(".svg") ||
      imagePath.toLowerCase().endsWith(".webp")) {
    return imagePath;
  }
  return imagePath.replace(/\.(jpg|jpeg|png|gif|bmp|tiff|tif)$/i, ".webp");
}

export function remarkFolderImages(options: { base?: string } = {}) {
  const base = options.base || '';
  return function transformer(tree: any, file: any) {
    visit(tree, "image", (node: any) => {
      if (!node.url || node.url.startsWith("/") || node.url.startsWith("http")) return;

      const urlLower = node.url.toLowerCase();
      const nonImageExtensions = [
        '.mp3', '.wav', '.ogg', '.m4a', '.3gp', '.flac', '.aac',
        '.mp4', '.webm', '.ogv', '.mov', '.mkv', '.avi',
        '.pdf'
      ];
      if (nonImageExtensions.some(ext => urlLower.endsWith(ext))) return;

      let collection: string | null = null;
      let contentSlug: string | null = null;
      let isFolderBased = false;

      if (file.path) {
        const normalizedPath = file.path.replace(/\\/g, "/");
        const pathParts = normalizedPath.split("/");

        if (normalizedPath.includes("/posts/")) {
          collection = "posts";
          const postsIndex = pathParts.indexOf("posts");
          isFolderBased = normalizedPath.endsWith("/index.md");
          contentSlug = isFolderBased ? pathParts[postsIndex + 1] : null;
        } else if (normalizedPath.includes("/projects/")) {
          collection = "projects";
          const projectsIndex = pathParts.indexOf("projects");
          isFolderBased = normalizedPath.endsWith("/index.md");
          contentSlug = isFolderBased ? pathParts[projectsIndex + 1] : null;
        } else if (normalizedPath.includes("/docs/")) {
          collection = "docs";
          const docsIndex = pathParts.indexOf("docs");
          isFolderBased = normalizedPath.endsWith("/index.md");
          contentSlug = isFolderBased ? pathParts[docsIndex + 1] : null;
        } else if (normalizedPath.includes("/pages/")) {
          collection = "pages";
          const pagesIndex = pathParts.indexOf("pages");
          isFolderBased = normalizedPath.endsWith("/index.md");
          contentSlug = isFolderBased ? pathParts[pagesIndex + 1] : null;
        } else if (normalizedPath.includes("/special/")) {
          collection = "pages";
          const specialIndex = pathParts.indexOf("special");
          isFolderBased = normalizedPath.endsWith("/index.md");
          contentSlug = isFolderBased ? pathParts[specialIndex + 1] : null;
        }
      }

      let imagePath = node.url;
      if (imagePath.startsWith("./")) imagePath = imagePath.slice(2);

      if (!collection && imagePath.startsWith("attachments/")) {
        collection = "pages";
      }
      if (!collection) return;

      if (isFolderBased && contentSlug) {
        let cleanImagePath = imagePath;
        if (cleanImagePath.startsWith('images/') || cleanImagePath.startsWith('attachments/')) {
          cleanImagePath = cleanImagePath.replace(/^(images|attachments)\//, '');
        }
        let finalUrl = withBase(base, `/${collection}/${contentSlug}/${cleanImagePath}`);
        finalUrl = convertToWebP(finalUrl);
        node.url = finalUrl;
      } else if (imagePath.startsWith("attachments/")) {
        let finalUrl = withBase(base, `/${collection}/${imagePath}`);
        finalUrl = convertToWebP(finalUrl);
        node.url = finalUrl;
      } else {
        let finalUrl = withBase(base, `/${collection}/attachments/${imagePath}`);
        finalUrl = convertToWebP(finalUrl);
        node.url = finalUrl;
      }

      if (node.data && node.data.hProperties) {
        node.data.hProperties.src = node.url;
      }
    });
  };
}

export function remarkImageCaptions() {
  return function transformer(tree: any) {
    visit(tree, "image", (node: any) => {
      if (node.title) {
        if (!node.data) node.data = {};
        if (!node.data.hProperties) node.data.hProperties = {};
        node.data.hProperties["data-caption"] = node.title;
        node.data.hProperties.title = node.title;
      }
    });
  };
}